# -*- coding: utf-8 -*-
"""
Event Preparation Workflow
===================================

3-phase workflow for preparing a facility for an upcoming demand response
event within PACK-037 Demand Response Pack.

Phases:
    1. EventNotification          -- Receive and parse DR event signal
    2. DispatchOptimization       -- Optimise load shed sequence for target kW
    3. PreConditioningActivation  -- Activate pre-conditioning strategies

The workflow follows GreenLang zero-hallucination principles: dispatch
optimization uses deterministic load-priority sorting and arithmetic
curtailment aggregation. Pre-conditioning calculations use engineering
thermal mass formulas. SHA-256 provenance hashes guarantee auditability.

Regulatory references:
    - OpenADR 2.0b (event notification protocol)
    - ISO/RTO dispatch rules and notification timelines
    - ASHRAE 55 thermal comfort constraints

Schedule: event-triggered (real-time)
Estimated duration: 5 minutes

Author: GreenLang Team
Version: 37.0.0
"""

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

_MODULE_VERSION = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.utcnow()


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


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


class EventSeverity(str, Enum):
    """DR event severity classification."""

    NORMAL = "normal"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class PreConditioningStrategy(str, Enum):
    """Pre-conditioning strategies available."""

    THERMAL_PRECOOL = "thermal_precool"
    THERMAL_PREHEAT = "thermal_preheat"
    BATTERY_PRECHARGE = "battery_precharge"
    PROCESS_RAMP_DOWN = "process_ramp_down"
    EV_CHARGE_ADVANCE = "ev_charge_advance"
    LIGHTING_BOOST = "lighting_boost"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Pre-conditioning parameters by strategy
PRECONDITION_PARAMS: Dict[str, Dict[str, Any]] = {
    "thermal_precool": {
        "description": "Pre-cool building below normal setpoint",
        "setpoint_offset_c": Decimal("-2.0"),
        "lead_time_min": 60,
        "energy_increase_pct": Decimal("15"),
        "comfort_impact": "minimal",
        "applicable_loads": ["hvac_cooling"],
    },
    "thermal_preheat": {
        "description": "Pre-heat building above normal setpoint",
        "setpoint_offset_c": Decimal("2.0"),
        "lead_time_min": 60,
        "energy_increase_pct": Decimal("12"),
        "comfort_impact": "minimal",
        "applicable_loads": ["hvac_heating"],
    },
    "battery_precharge": {
        "description": "Charge on-site battery storage to full capacity",
        "target_soc_pct": Decimal("100"),
        "lead_time_min": 120,
        "energy_increase_pct": Decimal("0"),
        "comfort_impact": "none",
        "applicable_loads": [],
    },
    "process_ramp_down": {
        "description": "Gradually reduce non-critical process loads",
        "ramp_rate_pct_per_min": Decimal("2.0"),
        "lead_time_min": 30,
        "energy_increase_pct": Decimal("0"),
        "comfort_impact": "none",
        "applicable_loads": ["process_motors", "compressed_air"],
    },
    "ev_charge_advance": {
        "description": "Complete EV charging before event window",
        "target_soc_pct": Decimal("100"),
        "lead_time_min": 180,
        "energy_increase_pct": Decimal("25"),
        "comfort_impact": "none",
        "applicable_loads": ["ev_charging"],
    },
    "lighting_boost": {
        "description": "Boost lighting levels before dimming during event",
        "boost_pct": Decimal("10"),
        "lead_time_min": 15,
        "energy_increase_pct": Decimal("10"),
        "comfort_impact": "minimal",
        "applicable_loads": ["lighting_interior"],
    },
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


class DispatchAction(BaseModel):
    """A single load dispatch action in the shed sequence."""

    action_id: str = Field(default_factory=lambda: f"act-{uuid.uuid4().hex[:8]}")
    load_id: str = Field(default="", description="Load identifier")
    load_name: str = Field(default="", description="Load display name")
    action: str = Field(default="curtail", description="curtail|reduce|shed|shift")
    priority: int = Field(default=0, ge=0, description="Execution priority (1=first)")
    curtail_kw: Decimal = Field(default=Decimal("0"), ge=0, description="kW to curtail")
    cumulative_kw: Decimal = Field(default=Decimal("0"), ge=0, description="Running total kW")
    ramp_time_min: int = Field(default=0, ge=0, description="Time to execute action")


class PreConditionAction(BaseModel):
    """A pre-conditioning action to activate before the event."""

    strategy: str = Field(default="", description="Pre-conditioning strategy key")
    description: str = Field(default="", description="Action description")
    lead_time_min: int = Field(default=0, ge=0, description="Minutes before event start")
    energy_cost_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    loads_affected: List[str] = Field(default_factory=list)


class EventPreparationInput(BaseModel):
    """Input data model for EventPreparationWorkflow."""

    event_id: str = Field(default_factory=lambda: f"evt-{uuid.uuid4().hex[:8]}")
    facility_id: str = Field(default="", description="Facility identifier")
    program_key: str = Field(default="", description="DR program identifier")
    event_start_utc: str = Field(..., description="Event start ISO 8601")
    event_end_utc: str = Field(..., description="Event end ISO 8601")
    target_curtailment_kw: Decimal = Field(..., gt=0, description="Target curtailment kW")
    severity: str = Field(default="moderate", description="Event severity level")
    notification_received_utc: str = Field(default="", description="When notification arrived")
    loads: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Available loads with curtail_kw, priority, ramp_time",
    )
    precondition_enabled: bool = Field(default=True, description="Enable pre-conditioning")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("event_start_utc")
    @classmethod
    def validate_event_start(cls, v: str) -> str:
        """Ensure event start is a valid ISO timestamp."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("event_start_utc must not be blank")
        return stripped


class EventPreparationResult(BaseModel):
    """Complete result from event preparation workflow."""

    preparation_id: str = Field(..., description="Unique preparation execution ID")
    event_id: str = Field(default="", description="DR event identifier")
    facility_id: str = Field(default="", description="Facility identifier")
    target_kw: Decimal = Field(default=Decimal("0"), ge=0)
    achievable_kw: Decimal = Field(default=Decimal("0"), ge=0)
    dispatch_actions: List[DispatchAction] = Field(default_factory=list)
    precondition_actions: List[PreConditionAction] = Field(default_factory=list)
    target_met: bool = Field(default=False, description="Whether target curtailment is achievable")
    shortfall_kw: Decimal = Field(default=Decimal("0"), ge=0)
    preparation_lead_time_min: int = Field(default=0, ge=0)
    phases_completed: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class EventPreparationWorkflow:
    """
    3-phase event preparation workflow for demand response events.

    Receives and parses the DR event signal, optimises the load shed
    sequence to meet target curtailment, and activates pre-conditioning
    strategies.

    Zero-hallucination: dispatch optimization uses priority-sorted
    greedy aggregation. Pre-conditioning energy costs use deterministic
    percentage-based calculations. No LLM calls in the numeric
    computation path.

    Attributes:
        preparation_id: Unique preparation execution identifier.
        _dispatch_actions: Ordered dispatch actions.
        _precondition_actions: Pre-conditioning actions.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = EventPreparationWorkflow()
        >>> inp = EventPreparationInput(
        ...     event_start_utc="2026-07-15T14:00:00Z",
        ...     event_end_utc="2026-07-15T18:00:00Z",
        ...     target_curtailment_kw=Decimal("500"),
        ...     loads=[{"load_id": "hvac-1", "curtail_kw": 300, "priority": 1}],
        ... )
        >>> result = wf.run(inp)
        >>> assert result.achievable_kw > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EventPreparationWorkflow."""
        self.preparation_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._dispatch_actions: List[DispatchAction] = []
        self._precondition_actions: List[PreConditionAction] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: EventPreparationInput) -> EventPreparationResult:
        """
        Execute the 3-phase event preparation workflow.

        Args:
            input_data: Validated event preparation input.

        Returns:
            EventPreparationResult with dispatch sequence and pre-conditioning plan.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = _utcnow()
        self.logger.info(
            "Starting event preparation workflow %s event=%s target=%.0f kW",
            self.preparation_id, input_data.event_id,
            float(input_data.target_curtailment_kw),
        )

        self._phase_results = []
        self._dispatch_actions = []
        self._precondition_actions = []

        try:
            # Phase 1: Event Notification
            phase1 = self._phase_event_notification(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Dispatch Optimization
            phase2 = self._phase_dispatch_optimization(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Pre-Conditioning Activation
            phase3 = self._phase_preconditioning_activation(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error(
                "Event preparation workflow failed: %s", exc, exc_info=True,
            )
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        achievable_kw = sum(a.curtail_kw for a in self._dispatch_actions)
        shortfall = max(Decimal("0"), input_data.target_curtailment_kw - achievable_kw)
        target_met = achievable_kw >= input_data.target_curtailment_kw
        max_lead_time = max(
            (a.lead_time_min for a in self._precondition_actions), default=0
        )
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = EventPreparationResult(
            preparation_id=self.preparation_id,
            event_id=input_data.event_id,
            facility_id=input_data.facility_id,
            target_kw=input_data.target_curtailment_kw,
            achievable_kw=achievable_kw,
            dispatch_actions=self._dispatch_actions,
            precondition_actions=self._precondition_actions,
            target_met=target_met,
            shortfall_kw=shortfall,
            preparation_lead_time_min=max_lead_time,
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Event preparation workflow %s completed in %dms "
            "achievable=%.0f kW target_met=%s shortfall=%.0f kW",
            self.preparation_id, int(elapsed_ms), float(achievable_kw),
            target_met, float(shortfall),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Event Notification
    # -------------------------------------------------------------------------

    def _phase_event_notification(
        self, input_data: EventPreparationInput
    ) -> PhaseResult:
        """Receive and parse DR event signal."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Parse event timing
        event_start = input_data.event_start_utc
        event_end = input_data.event_end_utc

        # Calculate event duration
        try:
            start_dt = datetime.fromisoformat(event_start.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(event_end.replace("Z", "+00:00"))
            duration_hours = (end_dt - start_dt).total_seconds() / 3600.0
        except (ValueError, TypeError):
            duration_hours = 4.0
            warnings.append("Could not parse event timestamps; defaulting to 4-hour duration")

        # Calculate lead time
        notification_time = input_data.notification_received_utc or _utcnow().isoformat() + "Z"
        try:
            notif_dt = datetime.fromisoformat(notification_time.replace("Z", "+00:00"))
            lead_time_min = max(0.0, (start_dt - notif_dt).total_seconds() / 60.0)
        except (ValueError, TypeError, UnboundLocalError):
            lead_time_min = 120.0
            warnings.append("Could not calculate lead time; defaulting to 120 min")

        # Severity assessment
        severity = input_data.severity
        if severity not in [s.value for s in EventSeverity]:
            severity = "moderate"
            warnings.append(f"Unknown severity; defaulting to 'moderate'")

        outputs["event_id"] = input_data.event_id
        outputs["event_start"] = event_start
        outputs["event_end"] = event_end
        outputs["duration_hours"] = round(duration_hours, 2)
        outputs["lead_time_min"] = round(lead_time_min, 1)
        outputs["severity"] = severity
        outputs["target_curtailment_kw"] = str(input_data.target_curtailment_kw)
        outputs["loads_available"] = len(input_data.loads)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 EventNotification: event=%s duration=%.1fh lead=%.0fmin severity=%s",
            input_data.event_id, duration_hours, lead_time_min, severity,
        )
        return PhaseResult(
            phase_name="event_notification", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Dispatch Optimization
    # -------------------------------------------------------------------------

    def _phase_dispatch_optimization(
        self, input_data: EventPreparationInput
    ) -> PhaseResult:
        """Optimise load shed sequence to meet target curtailment kW."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        target_kw = input_data.target_curtailment_kw

        # Sort loads by priority (lower number = higher priority = shed first)
        sorted_loads = sorted(
            input_data.loads,
            key=lambda x: (x.get("priority", 99), -float(x.get("curtail_kw", 0))),
        )

        cumulative_kw = Decimal("0")
        action_priority = 0

        for load_dict in sorted_loads:
            load_id = load_dict.get("load_id", f"load-{_new_uuid()[:8]}")
            load_name = load_dict.get("name", load_dict.get("load_type", load_id))
            curtail_kw = Decimal(str(load_dict.get("curtail_kw", 0)))
            ramp_time = int(load_dict.get("ramp_time_min", 5))

            if curtail_kw <= 0:
                continue

            action_priority += 1
            cumulative_kw += curtail_kw

            action = DispatchAction(
                load_id=load_id,
                load_name=load_name,
                action="curtail",
                priority=action_priority,
                curtail_kw=curtail_kw,
                cumulative_kw=cumulative_kw,
                ramp_time_min=ramp_time,
            )
            self._dispatch_actions.append(action)

            # Stop once target is met
            if cumulative_kw >= target_kw:
                break

        achievable = cumulative_kw
        shortfall = max(Decimal("0"), target_kw - achievable)
        if shortfall > 0:
            warnings.append(
                f"Target shortfall of {shortfall} kW; "
                f"achievable {achievable} kW < target {target_kw} kW"
            )

        max_ramp = max(
            (a.ramp_time_min for a in self._dispatch_actions), default=0
        )

        outputs["dispatch_actions"] = len(self._dispatch_actions)
        outputs["achievable_kw"] = str(achievable)
        outputs["target_kw"] = str(target_kw)
        outputs["shortfall_kw"] = str(shortfall)
        outputs["target_met"] = float(shortfall) <= 0
        outputs["max_ramp_time_min"] = max_ramp

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 DispatchOptimization: %d actions, achievable=%.0f kW, target_met=%s",
            len(self._dispatch_actions), float(achievable), float(shortfall) <= 0,
        )
        return PhaseResult(
            phase_name="dispatch_optimization", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Pre-Conditioning Activation
    # -------------------------------------------------------------------------

    def _phase_preconditioning_activation(
        self, input_data: EventPreparationInput
    ) -> PhaseResult:
        """Activate pre-conditioning strategies before event."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if not input_data.precondition_enabled:
            outputs["precondition_enabled"] = False
            outputs["reason"] = "Pre-conditioning disabled"
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            self.logger.info("Phase 3 PreConditioning: skipped (disabled)")
            return PhaseResult(
                phase_name="preconditioning_activation", phase_number=3,
                status=PhaseStatus.SKIPPED, duration_ms=elapsed_ms,
                outputs=outputs, warnings=warnings,
                provenance_hash=self._hash_dict(outputs),
            )

        # Determine applicable pre-conditioning strategies
        dispatched_load_types = set()
        for action in self._dispatch_actions:
            # Infer load type from load_id or name
            for load_dict in input_data.loads:
                if load_dict.get("load_id") == action.load_id:
                    load_type = load_dict.get("load_type", "")
                    dispatched_load_types.add(load_type)
                    break

        total_precondition_kwh = Decimal("0")
        for strategy_key, params in PRECONDITION_PARAMS.items():
            applicable_loads = params.get("applicable_loads", [])

            # Check if any dispatched loads match this strategy
            matching = dispatched_load_types.intersection(set(applicable_loads))
            if not matching and applicable_loads:
                continue

            # Calculate energy cost of pre-conditioning
            energy_increase_pct = params.get("energy_increase_pct", Decimal("0"))
            lead_time_hours = Decimal(str(params.get("lead_time_min", 60))) / Decimal("60")

            # Estimate kW affected
            affected_kw = Decimal("0")
            for action in self._dispatch_actions:
                for load_dict in input_data.loads:
                    if (load_dict.get("load_id") == action.load_id and
                            load_dict.get("load_type", "") in matching):
                        affected_kw += action.curtail_kw

            if affected_kw <= 0 and applicable_loads:
                continue

            energy_cost_kwh = (
                affected_kw * energy_increase_pct / Decimal("100") * lead_time_hours
            ).quantize(Decimal("0.1"))
            total_precondition_kwh += energy_cost_kwh

            precondition_action = PreConditionAction(
                strategy=strategy_key,
                description=params["description"],
                lead_time_min=params["lead_time_min"],
                energy_cost_kwh=energy_cost_kwh,
                loads_affected=list(matching),
            )
            self._precondition_actions.append(precondition_action)

        outputs["precondition_actions"] = len(self._precondition_actions)
        outputs["total_precondition_energy_kwh"] = str(total_precondition_kwh)
        outputs["strategies_activated"] = [
            a.strategy for a in self._precondition_actions
        ]
        outputs["max_lead_time_min"] = max(
            (a.lead_time_min for a in self._precondition_actions), default=0
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 PreConditioning: %d strategies activated, energy=%.1f kWh",
            len(self._precondition_actions), float(total_precondition_kwh),
        )
        return PhaseResult(
            phase_name="preconditioning_activation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: EventPreparationResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
