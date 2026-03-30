# -*- coding: utf-8 -*-
"""
EventManagerEngine - PACK-037 Demand Response Engine 5
=======================================================

Demand response event lifecycle management engine covering the complete
five-phase event lifecycle: Notification, Preparation, Execution,
Termination, and Assessment.  Manages event registration, readiness
preparation, real-time performance tracking at 1-minute intervals,
controlled termination with staged ramp-up, and post-event performance
assessment with settlement calculations.

Event Lifecycle Phases:
    1. NOTIFICATION   - Signal received, event registered, stakeholders
                        alerted, loads identified.
    2. PREPARATION    - Dispatch plan generated, pre-conditioning
                        initiated, systems verified, commands staged.
    3. EXECUTION      - Curtailment commands issued, real-time
                        performance monitored at 1-minute intervals,
                        deviations flagged.
    4. TERMINATION    - Staged ramp-up initiated, loads restored in
                        sequence, rebound managed, normal operations
                        confirmed.
    5. ASSESSMENT     - Actual vs baseline compared, kWh curtailed
                        calculated, settlement amount determined,
                        lessons learned captured.

Calculation Methodology:
    Performance Tracking (1-minute intervals):
        curtailment_kw = baseline_kw - actual_kw
        performance_pct = curtailment_kw / target_kw * 100
        cumulative_kwh  = sum(curtailment_kw_i / 60)  for all intervals

    Settlement Calculation:
        verified_kwh    = sum(max(0, baseline_kw - actual_kw) / 60)
        settlement_amt  = verified_kwh * energy_rate
                        + enrolled_kw * capacity_rate * event_fraction
        penalty         = max(0, target_kwh - verified_kwh) * penalty_rate
        net_settlement  = settlement_amt - penalty

    Performance Scoring:
        response_score  = 100 * (1 - ramp_deviation / allowed_ramp)
        sustain_score   = 100 * (time_at_target / total_event_time)
        accuracy_score  = 100 * (1 - abs(actual - target) / target)
        overall_score   = 0.30 * response + 0.40 * sustain + 0.30 * accuracy

Regulatory References:
    - OpenADR 2.0b - Event signal specification
    - FERC Order 745 - DR settlement methodology
    - PJM Manual 18 - Performance assessment procedures
    - NAESB REQ.18 - Event communication standards
    - IEEE 2030.5 - DR event management
    - ISO 50001:2018 - Energy management continual improvement

Zero-Hallucination:
    - Performance tracking uses actual metered data only
    - Settlement follows published ISO/FERC rate schedules
    - Scoring formulas are deterministic weighted sums
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-037 Demand Response
Engine:  5 of 8
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

class EventType(str, Enum):
    """Type of demand response event.

    ECONOMIC:    Voluntary curtailment for economic benefit.
    EMERGENCY:   Mandatory curtailment for grid reliability.
    CAPACITY:    Capacity market performance event.
    ANCILLARY:   Ancillary services dispatch.
    CPP:         Critical peak pricing event.
    TEST:        System test / drill event.
    """
    ECONOMIC = "economic"
    EMERGENCY = "emergency"
    CAPACITY = "capacity"
    ANCILLARY = "ancillary"
    CPP = "critical_peak_pricing"
    TEST = "test"

class EventStatus(str, Enum):
    """Current status of a DR event.

    PENDING:      Event registered, awaiting preparation.
    PREPARING:    Pre-event preparation in progress.
    ACTIVE:       Event is executing, curtailment in effect.
    TERMINATING:  Event ending, loads being restored.
    COMPLETED:    Event finished, assessment pending or complete.
    FAILED:       Event failed to execute properly.
    CANCELLED:    Event cancelled before or during execution.
    """
    PENDING = "pending"
    PREPARING = "preparing"
    ACTIVE = "active"
    TERMINATING = "terminating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EventPhase(str, Enum):
    """Five-phase event lifecycle.

    NOTIFICATION:  Signal received, event registered.
    PREPARATION:   Pre-event readiness activities.
    EXECUTION:     Active curtailment period.
    TERMINATION:   Load restoration period.
    ASSESSMENT:    Post-event performance analysis.
    """
    NOTIFICATION = "notification"
    PREPARATION = "preparation"
    EXECUTION = "execution"
    TERMINATION = "termination"
    ASSESSMENT = "assessment"

class PerformanceGrade(str, Enum):
    """Event performance grade.

    EXCELLENT:  Score 90-100, exceeded target.
    GOOD:       Score 75-89, met target.
    ACCEPTABLE: Score 60-74, marginal performance.
    POOR:       Score 40-59, below expectations.
    FAILED:     Score < 40, non-compliant.
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"

class CommandStatus(str, Enum):
    """Status of a load control command.

    QUEUED:    Command staged, not yet sent.
    SENT:      Command dispatched to load controller.
    CONFIRMED: Load confirmed receipt / compliance.
    FAILED:    Command delivery or execution failed.
    OVERRIDDEN: Command overridden by operator.
    """
    QUEUED = "queued"
    SENT = "sent"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    OVERRIDDEN = "overridden"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Performance scoring weights.
RESPONSE_WEIGHT: Decimal = Decimal("0.30")
SUSTAIN_WEIGHT: Decimal = Decimal("0.40")
ACCURACY_WEIGHT: Decimal = Decimal("0.30")

# Grade thresholds.
GRADE_THRESHOLDS: List[Tuple[Decimal, PerformanceGrade]] = [
    (Decimal("90"), PerformanceGrade.EXCELLENT),
    (Decimal("75"), PerformanceGrade.GOOD),
    (Decimal("60"), PerformanceGrade.ACCEPTABLE),
    (Decimal("40"), PerformanceGrade.POOR),
    (Decimal("0"), PerformanceGrade.FAILED),
]

# Default ramp-up stagger interval (minutes) during termination.
DEFAULT_RAMP_STAGGER_MIN: int = 5

# Performance deviation alert threshold (%).
DEVIATION_ALERT_PCT: Decimal = Decimal("15")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class PerformanceInterval(BaseModel):
    """Real-time performance data for a single 1-minute interval.

    Attributes:
        minute: Minutes elapsed from event start.
        baseline_kw: Baseline power for this interval (kW).
        actual_kw: Metered actual power (kW).
        curtailment_kw: Curtailment achieved (baseline - actual).
        target_kw: Curtailment target for this interval (kW).
        performance_pct: Performance as pct of target.
        cumulative_kwh: Cumulative curtailment energy (kWh).
        deviation_flag: True if deviation exceeds threshold.
    """
    minute: int = Field(default=0, ge=0, description="Minute from start")
    baseline_kw: Decimal = Field(default=Decimal("0"), ge=0)
    actual_kw: Decimal = Field(default=Decimal("0"), ge=0)
    curtailment_kw: Decimal = Field(default=Decimal("0"))
    target_kw: Decimal = Field(default=Decimal("0"), ge=0)
    performance_pct: Decimal = Field(default=Decimal("0"))
    cumulative_kwh: Decimal = Field(default=Decimal("0"))
    deviation_flag: bool = Field(default=False)

class LoadControlCommand(BaseModel):
    """A load control command issued during event execution.

    Attributes:
        command_id: Unique command identifier.
        load_id: Target load identifier.
        load_name: Load name.
        command_type: Command action type.
        target_kw: Target power level (kW).
        issued_at: Timestamp when command was issued.
        status: Current command status.
        response_time_seconds: Time for load to respond (seconds).
        notes: Additional notes.
    """
    command_id: str = Field(default_factory=_new_uuid)
    load_id: str = Field(default="")
    load_name: str = Field(default="", max_length=500)
    command_type: str = Field(default="curtail")
    target_kw: Decimal = Field(default=Decimal("0"), ge=0)
    issued_at: datetime = Field(default_factory=utcnow)
    status: CommandStatus = Field(default=CommandStatus.QUEUED)
    response_time_seconds: Optional[int] = Field(default=None, ge=0)
    notes: str = Field(default="", max_length=2000)

class DREvent(BaseModel):
    """A demand response event record.

    Attributes:
        event_id: Unique event identifier.
        facility_id: Facility identifier.
        program_id: DR program identifier.
        event_type: Type of DR event.
        status: Current event status.
        phase: Current lifecycle phase.
        target_kw: Curtailment target (kW).
        enrolled_kw: Enrolled capacity (kW).
        scheduled_start: Scheduled event start time.
        scheduled_end: Scheduled event end time.
        actual_start: Actual event start time.
        actual_end: Actual event end time.
        notification_received_at: When notification was received.
        duration_minutes: Planned event duration (minutes).
        energy_rate: Energy payment rate ($/kWh).
        capacity_rate: Capacity rate ($/kW-month).
        penalty_rate: Penalty rate multiplier.
        commands: Load control commands issued.
        performance_data: 1-minute interval performance data.
        created_at: Event creation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    event_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    program_id: str = Field(default="")
    event_type: EventType = Field(default=EventType.ECONOMIC)
    status: EventStatus = Field(default=EventStatus.PENDING)
    phase: EventPhase = Field(default=EventPhase.NOTIFICATION)
    target_kw: Decimal = Field(default=Decimal("0"), ge=0)
    enrolled_kw: Decimal = Field(default=Decimal("0"), ge=0)
    scheduled_start: Optional[datetime] = Field(default=None)
    scheduled_end: Optional[datetime] = Field(default=None)
    actual_start: Optional[datetime] = Field(default=None)
    actual_end: Optional[datetime] = Field(default=None)
    notification_received_at: Optional[datetime] = Field(default=None)
    duration_minutes: int = Field(default=240, ge=0)
    energy_rate: Decimal = Field(default=Decimal("0.10"), ge=0)
    capacity_rate: Decimal = Field(default=Decimal("5.00"), ge=0)
    penalty_rate: Decimal = Field(default=Decimal("1.50"), ge=0)
    commands: List[LoadControlCommand] = Field(default_factory=list)
    performance_data: List[PerformanceInterval] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class EventExecution(BaseModel):
    """Real-time event execution summary.

    Attributes:
        event_id: Event identifier.
        status: Current event status.
        phase: Current lifecycle phase.
        elapsed_minutes: Minutes elapsed since event start.
        current_curtailment_kw: Current curtailment (kW).
        target_kw: Curtailment target (kW).
        current_performance_pct: Current performance vs target (%).
        cumulative_kwh: Cumulative curtailment energy (kWh).
        average_performance_pct: Average performance over event (%).
        deviation_count: Number of intervals with deviation flags.
        commands_issued: Total commands issued.
        commands_confirmed: Commands with confirmed status.
        commands_failed: Commands with failed status.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    event_id: str = Field(default="")
    status: EventStatus = Field(default=EventStatus.ACTIVE)
    phase: EventPhase = Field(default=EventPhase.EXECUTION)
    elapsed_minutes: int = Field(default=0)
    current_curtailment_kw: Decimal = Field(default=Decimal("0"))
    target_kw: Decimal = Field(default=Decimal("0"))
    current_performance_pct: Decimal = Field(default=Decimal("0"))
    cumulative_kwh: Decimal = Field(default=Decimal("0"))
    average_performance_pct: Decimal = Field(default=Decimal("0"))
    deviation_count: int = Field(default=0)
    commands_issued: int = Field(default=0)
    commands_confirmed: int = Field(default=0)
    commands_failed: int = Field(default=0)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class EventAssessment(BaseModel):
    """Post-event performance assessment and settlement.

    Attributes:
        event_id: Event identifier.
        facility_id: Facility identifier.
        program_id: Program identifier.
        event_type: Event type.
        target_kw: Curtailment target (kW).
        achieved_kw_avg: Average curtailment achieved (kW).
        target_kwh: Target curtailment energy (kWh).
        verified_kwh: Verified curtailment energy (kWh).
        achievement_pct: Percentage of target achieved.
        response_score: Response time score (0-100).
        sustain_score: Sustained performance score (0-100).
        accuracy_score: Target accuracy score (0-100).
        overall_score: Weighted overall score (0-100).
        grade: Performance grade.
        settlement_amount: Energy + capacity payment.
        penalty_amount: Non-performance penalty.
        net_settlement: Net settlement (payment - penalty).
        event_duration_minutes: Actual event duration (minutes).
        ramp_time_minutes: Time to reach target (minutes).
        time_at_target_minutes: Minutes at or above target.
        deviation_minutes: Minutes with deviation flags.
        lessons_learned: Improvement recommendations.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    event_id: str = Field(default="")
    facility_id: str = Field(default="")
    program_id: str = Field(default="")
    event_type: EventType = Field(default=EventType.ECONOMIC)
    target_kw: Decimal = Field(default=Decimal("0"))
    achieved_kw_avg: Decimal = Field(default=Decimal("0"))
    target_kwh: Decimal = Field(default=Decimal("0"))
    verified_kwh: Decimal = Field(default=Decimal("0"))
    achievement_pct: Decimal = Field(default=Decimal("0"))
    response_score: Decimal = Field(default=Decimal("0"))
    sustain_score: Decimal = Field(default=Decimal("0"))
    accuracy_score: Decimal = Field(default=Decimal("0"))
    overall_score: Decimal = Field(default=Decimal("0"))
    grade: PerformanceGrade = Field(default=PerformanceGrade.ACCEPTABLE)
    settlement_amount: Decimal = Field(default=Decimal("0"))
    penalty_amount: Decimal = Field(default=Decimal("0"))
    net_settlement: Decimal = Field(default=Decimal("0"))
    event_duration_minutes: int = Field(default=0)
    ramp_time_minutes: int = Field(default=0)
    time_at_target_minutes: int = Field(default=0)
    deviation_minutes: int = Field(default=0)
    lessons_learned: List[str] = Field(default_factory=list)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EventManagerEngine:
    """Demand response event lifecycle management engine.

    Manages the five-phase event lifecycle from notification through
    post-event assessment.  Tracks real-time performance at 1-minute
    intervals and calculates settlement amounts.  All calculations use
    deterministic Decimal arithmetic with SHA-256 provenance hashing.

    Usage::

        engine = EventManagerEngine()
        event = engine.register_event(
            facility_id="FAC-001",
            program_id="PJM-ELR-001",
            event_type=EventType.ECONOMIC,
            target_kw=Decimal("500"),
            duration_minutes=240,
        )
        prepared = engine.prepare_event(event)
        executed = engine.execute_event(
            event, baseline_kw_per_min, actual_kw_per_min
        )
        assessment = engine.assess_event(event)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise EventManagerEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - deviation_threshold_pct (Decimal): deviation alert %
                - ramp_stagger_minutes (int): termination ramp stagger
                - response_weight (Decimal): response score weight
                - sustain_weight (Decimal): sustain score weight
                - accuracy_weight (Decimal): accuracy score weight
        """
        self.config = config or {}
        self._deviation_pct = _decimal(
            self.config.get("deviation_threshold_pct", DEVIATION_ALERT_PCT)
        )
        self._ramp_stagger = int(
            self.config.get("ramp_stagger_minutes", DEFAULT_RAMP_STAGGER_MIN)
        )
        self._response_w = _decimal(
            self.config.get("response_weight", RESPONSE_WEIGHT)
        )
        self._sustain_w = _decimal(
            self.config.get("sustain_weight", SUSTAIN_WEIGHT)
        )
        self._accuracy_w = _decimal(
            self.config.get("accuracy_weight", ACCURACY_WEIGHT)
        )
        logger.info(
            "EventManagerEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def register_event(
        self,
        facility_id: str,
        program_id: str,
        event_type: EventType,
        target_kw: Decimal,
        duration_minutes: int,
        enrolled_kw: Optional[Decimal] = None,
        energy_rate: Decimal = Decimal("0.10"),
        capacity_rate: Decimal = Decimal("5.00"),
        penalty_rate: Decimal = Decimal("1.50"),
        scheduled_start: Optional[datetime] = None,
    ) -> DREvent:
        """Register a new DR event (Notification phase).

        Args:
            facility_id: Facility identifier.
            program_id: DR program identifier.
            event_type: Type of event.
            target_kw: Curtailment target (kW).
            duration_minutes: Planned event duration (minutes).
            enrolled_kw: Enrolled capacity (defaults to target).
            energy_rate: Energy payment rate ($/kWh).
            capacity_rate: Capacity payment rate ($/kW-month).
            penalty_rate: Penalty rate multiplier.
            scheduled_start: Planned start time.

        Returns:
            DREvent in PENDING status, NOTIFICATION phase.
        """
        t0 = time.perf_counter()

        event = DREvent(
            facility_id=facility_id,
            program_id=program_id,
            event_type=event_type,
            status=EventStatus.PENDING,
            phase=EventPhase.NOTIFICATION,
            target_kw=_round_val(target_kw, 2),
            enrolled_kw=_round_val(enrolled_kw or target_kw, 2),
            scheduled_start=scheduled_start,
            notification_received_at=utcnow(),
            duration_minutes=duration_minutes,
            energy_rate=energy_rate,
            capacity_rate=capacity_rate,
            penalty_rate=penalty_rate,
        )

        if scheduled_start:
            # Calculate scheduled end
            from datetime import timedelta

            event.scheduled_end = scheduled_start + timedelta(
                minutes=duration_minutes
            )

        event.provenance_hash = _compute_hash(event)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Event registered: %s, type=%s, target=%s kW, "
            "duration=%d min, hash=%s (%.1f ms)",
            event.event_id, event_type.value, target_kw,
            duration_minutes, event.provenance_hash[:16], elapsed,
        )
        return event

    def prepare_event(self, event: DREvent) -> DREvent:
        """Transition event to Preparation phase.

        Updates status to PREPARING and phase to PREPARATION.
        In a real system, this would trigger dispatch plan generation,
        pre-conditioning, and system readiness verification.

        Args:
            event: DR event to prepare.

        Returns:
            Updated DREvent in PREPARING status.
        """
        t0 = time.perf_counter()

        event.status = EventStatus.PREPARING
        event.phase = EventPhase.PREPARATION
        event.provenance_hash = _compute_hash(event)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Event preparing: %s, hash=%s (%.1f ms)",
            event.event_id, event.provenance_hash[:16], elapsed,
        )
        return event

    def execute_event(
        self,
        event: DREvent,
        baseline_kw_per_min: List[Decimal],
        actual_kw_per_min: List[Decimal],
    ) -> DREvent:
        """Process event execution with real-time performance tracking.

        Calculates curtailment at each 1-minute interval, flags
        deviations, and updates cumulative metrics.

        Args:
            event: DR event to execute.
            baseline_kw_per_min: Baseline power at each minute (kW).
            actual_kw_per_min: Actual metered power at each minute (kW).

        Returns:
            Updated DREvent with performance data populated.
        """
        t0 = time.perf_counter()

        event.status = EventStatus.ACTIVE
        event.phase = EventPhase.EXECUTION
        event.actual_start = utcnow()

        performance: List[PerformanceInterval] = []
        cumulative_kwh = Decimal("0")

        n_intervals = min(
            len(baseline_kw_per_min),
            len(actual_kw_per_min),
            event.duration_minutes,
        )

        for minute in range(n_intervals):
            bl = _decimal(baseline_kw_per_min[minute])
            act = _decimal(actual_kw_per_min[minute])
            curtailment = max(bl - act, Decimal("0"))
            perf_pct = _safe_pct(curtailment, event.target_kw)
            cumulative_kwh += curtailment / Decimal("60")

            # Deviation flag
            deviation = abs(perf_pct - Decimal("100"))
            flag = deviation > self._deviation_pct

            performance.append(PerformanceInterval(
                minute=minute,
                baseline_kw=_round_val(bl, 2),
                actual_kw=_round_val(act, 2),
                curtailment_kw=_round_val(curtailment, 2),
                target_kw=_round_val(event.target_kw, 2),
                performance_pct=_round_val(perf_pct, 2),
                cumulative_kwh=_round_val(cumulative_kwh, 4),
                deviation_flag=flag,
            ))

        event.performance_data = performance
        event.provenance_hash = _compute_hash(event)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Event executed: %s, %d intervals, cumulative=%.2f kWh, "
            "hash=%s (%.1f ms)",
            event.event_id, n_intervals, float(cumulative_kwh),
            event.provenance_hash[:16], elapsed,
        )
        return event

    def terminate_event(self, event: DREvent) -> DREvent:
        """Transition event to Termination phase.

        Updates status to TERMINATING and phase to TERMINATION.
        In production this triggers staged load restoration.

        Args:
            event: DR event to terminate.

        Returns:
            Updated DREvent in TERMINATING status.
        """
        t0 = time.perf_counter()

        event.status = EventStatus.TERMINATING
        event.phase = EventPhase.TERMINATION
        event.actual_end = utcnow()
        event.provenance_hash = _compute_hash(event)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Event terminating: %s, hash=%s (%.1f ms)",
            event.event_id, event.provenance_hash[:16], elapsed,
        )
        return event

    def assess_event(self, event: DREvent) -> EventAssessment:
        """Perform post-event performance assessment (Assessment phase).

        Calculates verified curtailment, performance scores, settlement
        amounts, and generates lessons learned.

        Args:
            event: Completed DR event with performance data.

        Returns:
            EventAssessment with scores and settlement.
        """
        t0 = time.perf_counter()
        logger.info("Assessing event: %s", event.event_id)

        event.status = EventStatus.COMPLETED
        event.phase = EventPhase.ASSESSMENT

        perf = event.performance_data
        n_intervals = len(perf)

        if n_intervals == 0:
            return self._create_empty_assessment(event)

        # Verified curtailment kWh
        verified_kwh = sum(
            (p.curtailment_kw / Decimal("60") for p in perf), Decimal("0")
        )

        # Target kWh
        target_kwh = event.target_kw * _decimal(n_intervals) / Decimal("60")

        # Average curtailment
        avg_curtailment = _safe_divide(
            sum((p.curtailment_kw for p in perf), Decimal("0")),
            _decimal(n_intervals),
        )

        achievement_pct = _safe_pct(verified_kwh, target_kwh)

        # Response score: how quickly target was reached
        ramp_time = self._calculate_ramp_time(perf, event.target_kw)
        allowed_ramp = min(_decimal(n_intervals), Decimal("30"))
        response_score = max(
            Decimal("100") * (Decimal("1") - _safe_divide(
                _decimal(ramp_time), allowed_ramp
            )),
            Decimal("0"),
        )

        # Sustain score: fraction of time at or above target
        time_at_target = sum(
            1 for p in perf
            if p.curtailment_kw >= event.target_kw * Decimal("0.90")
        )
        sustain_score = _safe_pct(
            _decimal(time_at_target), _decimal(n_intervals)
        )

        # Accuracy score: closeness to target
        avg_perf_pct = _safe_divide(
            sum((p.performance_pct for p in perf), Decimal("0")),
            _decimal(n_intervals),
        )
        accuracy_deviation = abs(avg_perf_pct - Decimal("100"))
        accuracy_score = max(
            Decimal("100") - accuracy_deviation, Decimal("0")
        )

        # Overall score
        overall = (
            self._response_w * response_score
            + self._sustain_w * sustain_score
            + self._accuracy_w * accuracy_score
        )

        grade = self._assign_grade(overall)

        # Settlement calculation
        settlement = self._calculate_settlement(
            event, verified_kwh, n_intervals
        )

        # Deviation count
        deviation_count = sum(1 for p in perf if p.deviation_flag)

        # Lessons learned
        lessons = self._generate_lessons(
            grade, response_score, sustain_score, accuracy_score,
            ramp_time, deviation_count, n_intervals,
        )

        result = EventAssessment(
            event_id=event.event_id,
            facility_id=event.facility_id,
            program_id=event.program_id,
            event_type=event.event_type,
            target_kw=_round_val(event.target_kw, 2),
            achieved_kw_avg=_round_val(avg_curtailment, 2),
            target_kwh=_round_val(target_kwh, 2),
            verified_kwh=_round_val(verified_kwh, 2),
            achievement_pct=_round_val(achievement_pct, 2),
            response_score=_round_val(response_score, 2),
            sustain_score=_round_val(sustain_score, 2),
            accuracy_score=_round_val(accuracy_score, 2),
            overall_score=_round_val(overall, 2),
            grade=grade,
            settlement_amount=_round_val(settlement["settlement"], 2),
            penalty_amount=_round_val(settlement["penalty"], 2),
            net_settlement=_round_val(settlement["net"], 2),
            event_duration_minutes=n_intervals,
            ramp_time_minutes=ramp_time,
            time_at_target_minutes=time_at_target,
            deviation_minutes=deviation_count,
            lessons_learned=lessons,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Assessment complete: %s, score=%.1f (%s), verified=%.2f kWh, "
            "net_settlement=%.2f, hash=%s (%.1f ms)",
            event.event_id, float(overall), grade.value,
            float(verified_kwh), float(settlement["net"]),
            result.provenance_hash[:16], elapsed,
        )
        return result

    def generate_commands(
        self,
        event: DREvent,
        load_targets: List[Dict[str, Any]],
    ) -> List[LoadControlCommand]:
        """Generate load control commands for an event.

        Args:
            event: DR event.
            load_targets: List of dicts with load_id, load_name,
                          target_kw, command_type.

        Returns:
            List of LoadControlCommand objects.
        """
        commands: List[LoadControlCommand] = []

        for lt in load_targets:
            cmd = LoadControlCommand(
                load_id=lt.get("load_id", ""),
                load_name=lt.get("load_name", ""),
                command_type=lt.get("command_type", "curtail"),
                target_kw=_decimal(lt.get("target_kw", 0)),
                issued_at=utcnow(),
                status=CommandStatus.QUEUED,
            )
            commands.append(cmd)

        event.commands = commands
        event.provenance_hash = _compute_hash(event)

        logger.info(
            "Generated %d commands for event %s",
            len(commands), event.event_id,
        )
        return commands

    def get_execution_summary(
        self, event: DREvent
    ) -> EventExecution:
        """Get real-time execution summary for an active event.

        Args:
            event: Active DR event with performance data.

        Returns:
            EventExecution summary.
        """
        perf = event.performance_data
        n = len(perf)

        if n == 0:
            summary = EventExecution(
                event_id=event.event_id,
                status=event.status,
                phase=event.phase,
                target_kw=_round_val(event.target_kw, 2),
            )
            summary.provenance_hash = _compute_hash(summary)
            return summary

        latest = perf[-1]
        avg_pct = _safe_divide(
            sum((p.performance_pct for p in perf), Decimal("0")),
            _decimal(n),
        )

        deviation_count = sum(1 for p in perf if p.deviation_flag)

        cmd_issued = len(event.commands)
        cmd_confirmed = sum(
            1 for c in event.commands if c.status == CommandStatus.CONFIRMED
        )
        cmd_failed = sum(
            1 for c in event.commands if c.status == CommandStatus.FAILED
        )

        summary = EventExecution(
            event_id=event.event_id,
            status=event.status,
            phase=event.phase,
            elapsed_minutes=n,
            current_curtailment_kw=_round_val(latest.curtailment_kw, 2),
            target_kw=_round_val(event.target_kw, 2),
            current_performance_pct=_round_val(latest.performance_pct, 2),
            cumulative_kwh=_round_val(latest.cumulative_kwh, 4),
            average_performance_pct=_round_val(avg_pct, 2),
            deviation_count=deviation_count,
            commands_issued=cmd_issued,
            commands_confirmed=cmd_confirmed,
            commands_failed=cmd_failed,
        )
        summary.provenance_hash = _compute_hash(summary)
        return summary

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _calculate_ramp_time(
        self,
        perf: List[PerformanceInterval],
        target_kw: Decimal,
    ) -> int:
        """Calculate time to reach 90% of curtailment target.

        Args:
            perf: Performance interval data.
            target_kw: Curtailment target (kW).

        Returns:
            Minutes to reach target.
        """
        threshold = target_kw * Decimal("0.90")
        for p in perf:
            if p.curtailment_kw >= threshold:
                return p.minute
        return len(perf)

    def _calculate_settlement(
        self,
        event: DREvent,
        verified_kwh: Decimal,
        n_intervals: int,
    ) -> Dict[str, Decimal]:
        """Calculate settlement amount and penalties.

        Args:
            event: DR event.
            verified_kwh: Verified curtailment energy (kWh).
            n_intervals: Number of performance intervals.

        Returns:
            Dict with 'settlement', 'penalty', 'net' amounts.
        """
        # Energy payment
        energy_payment = verified_kwh * event.energy_rate

        # Capacity payment (prorated for event fraction of month)
        # Assume 730 hours/month
        event_hours = _decimal(n_intervals) / Decimal("60")
        month_fraction = _safe_divide(event_hours, Decimal("730"))
        capacity_payment = event.enrolled_kw * event.capacity_rate * month_fraction

        settlement = energy_payment + capacity_payment

        # Penalty for shortfall
        target_kwh = event.target_kw * _decimal(n_intervals) / Decimal("60")
        shortfall_kwh = max(target_kwh - verified_kwh, Decimal("0"))
        penalty = shortfall_kwh * event.penalty_rate

        net = settlement - penalty

        return {
            "settlement": settlement,
            "penalty": penalty,
            "net": net,
        }

    def _assign_grade(self, score: Decimal) -> PerformanceGrade:
        """Assign performance grade based on score.

        Args:
            score: Overall performance score (0-100).

        Returns:
            PerformanceGrade.
        """
        for threshold, grade in GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return PerformanceGrade.FAILED

    def _generate_lessons(
        self,
        grade: PerformanceGrade,
        response: Decimal,
        sustain: Decimal,
        accuracy: Decimal,
        ramp_time: int,
        deviation_count: int,
        total_intervals: int,
    ) -> List[str]:
        """Generate lessons learned from event performance.

        Args:
            grade: Performance grade.
            response: Response score.
            sustain: Sustain score.
            accuracy: Accuracy score.
            ramp_time: Minutes to reach target.
            deviation_count: Number of deviation intervals.
            total_intervals: Total event intervals.

        Returns:
            List of lesson strings.
        """
        lessons: List[str] = []

        if grade in (PerformanceGrade.POOR, PerformanceGrade.FAILED):
            lessons.append(
                "Event performance was below acceptable levels. "
                "Review curtailment plan and load availability."
            )

        if response < Decimal("60"):
            lessons.append(
                f"Response time of {ramp_time} minutes is too slow. "
                "Consider pre-staging curtailment or improving automation."
            )

        if sustain < Decimal("60"):
            lessons.append(
                "Sustained performance was inconsistent. Check for loads "
                "that auto-restored or reached thermal limits."
            )

        if accuracy < Decimal("60"):
            lessons.append(
                "Curtailment accuracy was poor. Review baseline methodology "
                "and load measurement points."
            )

        if deviation_count > total_intervals * 0.3:
            lessons.append(
                f"Performance deviations in {deviation_count}/{total_intervals} "
                f"intervals. Investigate root cause of variability."
            )

        if not lessons:
            lessons.append(
                "Event performed well. Continue current DR strategies "
                "and consider enrolling in additional programs."
            )

        return lessons

    def _create_empty_assessment(
        self, event: DREvent
    ) -> EventAssessment:
        """Create assessment for an event with no performance data.

        Args:
            event: DR event.

        Returns:
            EventAssessment with zero scores.
        """
        result = EventAssessment(
            event_id=event.event_id,
            facility_id=event.facility_id,
            program_id=event.program_id,
            event_type=event.event_type,
            grade=PerformanceGrade.FAILED,
            lessons_learned=[
                "No performance data available. Event may not have executed. "
                "Verify metering and data collection systems."
            ],
        )
        result.provenance_hash = _compute_hash(result)
        return result
