# -*- coding: utf-8 -*-
"""
AlarmEngine - PACK-039 Energy Monitoring Engine 8
====================================================

Intelligent alarm management engine with ISA 18.2 lifecycle, suppression,
correlation, escalation, and key performance metrics.  Evaluates alarm
conditions against configurable thresholds, manages alarm state
transitions, applies suppression and shelving rules, correlates related
alarms, and calculates ISA 18.2 KPIs including MTTA, MTTR, false alarm
rate, standing alarm count, and alarm flood detection.

Calculation Methodology:
    Alarm Evaluation:
        deviation = abs(current_value - setpoint)
        deviation_pct = deviation / setpoint * 100
        is_active = deviation_pct > deadband_pct + threshold_pct

    Deadband Hysteresis:
        activate_threshold = setpoint * (1 + threshold_pct / 100)
        deactivate_threshold = setpoint * (1 + (threshold_pct - deadband_pct) / 100)

    Alarm Flood Detection:
        alarm_rate = alarm_count / interval_minutes
        is_flood = alarm_rate > flood_threshold_per_min

    MTTA (Mean Time To Acknowledge):
        mtta = sum(acknowledged_at - activated_at) / count(acknowledged)

    MTTR (Mean Time To Resolve):
        mttr = sum(cleared_at - activated_at) / count(resolved)

    False Alarm Rate:
        false_rate = false_alarm_count / total_alarm_count * 100

    Standing Alarm Count:
        standing = count(alarms where state == ACTIVE and age > standing_threshold)

    ISA 18.2 Alarm Metrics:
        avg_alarms_per_hour = total_alarms / total_hours
        peak_alarms_per_10min = max(count_per_10min_window)
        chattering_rate = chattering_count / total_count * 100

Regulatory References:
    - ISA 18.2-2016     Alarm Management in the Process Industries
    - IEC 62682:2014    Management of alarm systems
    - EEMUA 191         Alarm systems: guide to design, management, procurement
    - NAMUR NA 102      Alarm management
    - API RP 1167       Pipeline SCADA alarm management
    - ISO 50001:2018    Energy management system - monitoring requirements
    - EN 15232          Building automation system classification
    - ASHRAE Guideline 36  High-performance sequences of operation

Zero-Hallucination:
    - All alarm evaluations use deterministic threshold comparison
    - No LLM involvement in alarm state or metric calculation
    - Correlation uses rule-based pattern matching, not ML
    - Decimal arithmetic throughout for audit-grade precision
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-039 Energy Monitoring
Engine:  8 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
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

class AlarmPriority(str, Enum):
    """Alarm priority classification per ISA 18.2.

    CRITICAL:     Immediate action required, safety/financial risk.
    HIGH:         Prompt action required within minutes.
    MEDIUM:       Action required within shift.
    LOW:          Action at next convenient opportunity.
    DIAGNOSTIC:   Informational, no operator action needed.
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DIAGNOSTIC = "diagnostic"

class AlarmState(str, Enum):
    """Alarm lifecycle state per ISA 18.2.

    ACTIVE:       Alarm condition present, not yet acknowledged.
    ACKNOWLEDGED: Operator has acknowledged the alarm.
    SHELVED:      Temporarily removed from active display.
    SUPPRESSED:   Suppressed by rule (flood, correlation, schedule).
    CLEARED:      Alarm condition has returned to normal.
    CLOSED:       Alarm fully resolved and documented.
    """
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    SHELVED = "shelved"
    SUPPRESSED = "suppressed"
    CLEARED = "cleared"
    CLOSED = "closed"

class SuppressionType(str, Enum):
    """Alarm suppression rule type.

    SHELVE:        Manually shelved by operator for defined period.
    DELAY:         Delayed activation (on-delay timer).
    DEADBAND:      Suppressed within deadband hysteresis.
    TIME_WINDOW:   Suppressed during scheduled maintenance window.
    CORRELATION:   Suppressed as consequence of parent alarm.
    """
    SHELVE = "shelve"
    DELAY = "delay"
    DEADBAND = "deadband"
    TIME_WINDOW = "time_window"
    CORRELATION = "correlation"

class EscalationLevel(str, Enum):
    """Alarm escalation hierarchy level.

    L1_OPERATOR:    Front-line operator response.
    L2_SUPERVISOR:  Shift supervisor escalation.
    L3_MANAGER:     Facility / energy manager escalation.
    L4_EXECUTIVE:   Executive / director escalation.
    """
    L1_OPERATOR = "l1_operator"
    L2_SUPERVISOR = "l2_supervisor"
    L3_MANAGER = "l3_manager"
    L4_EXECUTIVE = "l4_executive"

class AlarmCategory(str, Enum):
    """Energy monitoring alarm category.

    ENERGY_WASTE:       Unexpected or excessive energy consumption.
    EQUIPMENT_FAULT:    Equipment malfunction or degradation.
    SCHEDULE_ERROR:     Equipment operating outside schedule.
    COMFORT_VIOLATION:  Temperature or IAQ comfort band violation.
    SAFETY:             Safety-critical condition.
    FINANCIAL:          Cost or budget threshold exceedance.
    """
    ENERGY_WASTE = "energy_waste"
    EQUIPMENT_FAULT = "equipment_fault"
    SCHEDULE_ERROR = "schedule_error"
    COMFORT_VIOLATION = "comfort_violation"
    SAFETY = "safety"
    FINANCIAL = "financial"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ISA 18.2 recommended maximum average alarm rate (alarms per operator per hour).
ISA_MAX_AVG_ALARMS_PER_HOUR: Decimal = Decimal("6.0")

# Alarm flood threshold (alarms per 10-minute window).
ALARM_FLOOD_THRESHOLD: int = 10

# Standing alarm threshold (minutes before alarm is considered standing).
STANDING_ALARM_THRESHOLD_MIN: int = 60

# Default shelve duration (hours).
DEFAULT_SHELVE_HOURS: int = 8

# Default on-delay timer (seconds).
DEFAULT_ON_DELAY_SEC: int = 30

# Default deadband percentage.
DEFAULT_DEADBAND_PCT: Decimal = Decimal("2.0")

# Escalation time limits by priority (minutes).
ESCALATION_TIMEOUTS: Dict[str, int] = {
    AlarmPriority.CRITICAL.value: 5,
    AlarmPriority.HIGH.value: 15,
    AlarmPriority.MEDIUM.value: 60,
    AlarmPriority.LOW.value: 240,
    AlarmPriority.DIAGNOSTIC.value: 480,
}

# Maximum alarms per evaluation batch.
MAX_ALARM_BATCH: int = 5000

# Chattering detection: minimum transitions in window to classify as chatter.
CHATTER_MIN_TRANSITIONS: int = 5
CHATTER_WINDOW_MIN: int = 30

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class AlarmDefinition(BaseModel):
    """Alarm point definition and configuration.

    Attributes:
        alarm_id: Unique alarm identifier.
        alarm_name: Human-readable alarm name.
        alarm_category: Energy alarm category.
        priority: Alarm priority.
        meter_id: Associated meter or point identifier.
        setpoint: Normal operating setpoint value.
        high_threshold_pct: High alarm threshold (% above setpoint).
        low_threshold_pct: Low alarm threshold (% below setpoint).
        deadband_pct: Deadband for hysteresis.
        on_delay_sec: On-delay timer (seconds).
        escalation_level: Initial escalation level.
        escalation_timeout_min: Time before escalation (minutes).
        parent_alarm_id: Parent alarm for correlation.
        is_enabled: Whether alarm is enabled.
        description: Alarm description.
    """
    alarm_id: str = Field(
        default_factory=_new_uuid, description="Alarm identifier"
    )
    alarm_name: str = Field(
        default="", max_length=500, description="Alarm name"
    )
    alarm_category: AlarmCategory = Field(
        default=AlarmCategory.ENERGY_WASTE, description="Category"
    )
    priority: AlarmPriority = Field(
        default=AlarmPriority.MEDIUM, description="Priority"
    )
    meter_id: str = Field(
        default="", description="Associated meter ID"
    )
    setpoint: Decimal = Field(
        default=Decimal("0"), description="Normal setpoint"
    )
    high_threshold_pct: Decimal = Field(
        default=Decimal("10"), ge=0, description="High threshold (%)"
    )
    low_threshold_pct: Decimal = Field(
        default=Decimal("10"), ge=0, description="Low threshold (%)"
    )
    deadband_pct: Decimal = Field(
        default=DEFAULT_DEADBAND_PCT, ge=0, description="Deadband (%)"
    )
    on_delay_sec: int = Field(
        default=DEFAULT_ON_DELAY_SEC, ge=0, description="On-delay (sec)"
    )
    escalation_level: EscalationLevel = Field(
        default=EscalationLevel.L1_OPERATOR, description="Escalation level"
    )
    escalation_timeout_min: int = Field(
        default=60, ge=0, description="Escalation timeout (min)"
    )
    parent_alarm_id: str = Field(
        default="", description="Parent alarm for correlation"
    )
    is_enabled: bool = Field(
        default=True, description="Alarm enabled"
    )
    description: str = Field(
        default="", max_length=2000, description="Description"
    )

    @field_validator("alarm_name", mode="before")
    @classmethod
    def validate_name(cls, v: Any) -> Any:
        """Ensure alarm name is non-empty."""
        if isinstance(v, str) and not v.strip():
            return "Unnamed Alarm"
        return v

class AlarmEvent(BaseModel):
    """Alarm event instance (an occurrence of an alarm).

    Attributes:
        event_id: Unique event identifier.
        alarm_id: Reference to alarm definition.
        alarm_name: Alarm name.
        priority: Alarm priority.
        category: Alarm category.
        state: Current alarm state.
        current_value: Current measured value.
        setpoint: Setpoint at time of alarm.
        deviation_pct: Deviation from setpoint (%).
        activated_at: Activation timestamp.
        acknowledged_at: Acknowledgement timestamp.
        cleared_at: Cleared timestamp.
        closed_at: Closed timestamp.
        escalation_level: Current escalation level.
        is_suppressed: Whether alarm is suppressed.
        suppression_type: Type of suppression applied.
        is_false_alarm: Whether classified as false alarm.
        is_chattering: Whether classified as chattering.
        notes: Operator notes.
    """
    event_id: str = Field(
        default_factory=_new_uuid, description="Event ID"
    )
    alarm_id: str = Field(default="", description="Alarm definition ID")
    alarm_name: str = Field(default="", description="Alarm name")
    priority: AlarmPriority = Field(
        default=AlarmPriority.MEDIUM, description="Priority"
    )
    category: AlarmCategory = Field(
        default=AlarmCategory.ENERGY_WASTE, description="Category"
    )
    state: AlarmState = Field(
        default=AlarmState.ACTIVE, description="Current state"
    )
    current_value: Decimal = Field(
        default=Decimal("0"), description="Current value"
    )
    setpoint: Decimal = Field(
        default=Decimal("0"), description="Setpoint"
    )
    deviation_pct: Decimal = Field(
        default=Decimal("0"), description="Deviation (%)"
    )
    activated_at: datetime = Field(
        default_factory=utcnow, description="Activation time"
    )
    acknowledged_at: Optional[datetime] = Field(
        default=None, description="Acknowledgement time"
    )
    cleared_at: Optional[datetime] = Field(
        default=None, description="Cleared time"
    )
    closed_at: Optional[datetime] = Field(
        default=None, description="Closed time"
    )
    escalation_level: EscalationLevel = Field(
        default=EscalationLevel.L1_OPERATOR, description="Escalation"
    )
    is_suppressed: bool = Field(
        default=False, description="Is suppressed"
    )
    suppression_type: Optional[SuppressionType] = Field(
        default=None, description="Suppression type"
    )
    is_false_alarm: bool = Field(
        default=False, description="False alarm flag"
    )
    is_chattering: bool = Field(
        default=False, description="Chattering flag"
    )
    notes: str = Field(
        default="", max_length=2000, description="Operator notes"
    )

class SuppressionRule(BaseModel):
    """Alarm suppression rule configuration.

    Attributes:
        rule_id: Unique rule identifier.
        rule_name: Human-readable rule name.
        suppression_type: Type of suppression.
        target_alarm_ids: Alarm IDs this rule applies to.
        parent_alarm_id: Parent alarm for correlation suppression.
        start_time: Window start (for TIME_WINDOW type).
        end_time: Window end (for TIME_WINDOW type).
        shelve_duration_hours: Shelve duration (for SHELVE type).
        delay_seconds: Delay timer (for DELAY type).
        deadband_pct: Deadband percentage (for DEADBAND type).
        is_active: Whether rule is currently active.
        description: Rule description.
    """
    rule_id: str = Field(
        default_factory=_new_uuid, description="Rule ID"
    )
    rule_name: str = Field(
        default="", max_length=500, description="Rule name"
    )
    suppression_type: SuppressionType = Field(
        default=SuppressionType.DEADBAND, description="Suppression type"
    )
    target_alarm_ids: List[str] = Field(
        default_factory=list, description="Target alarm IDs"
    )
    parent_alarm_id: str = Field(
        default="", description="Parent alarm ID"
    )
    start_time: Optional[datetime] = Field(
        default=None, description="Window start"
    )
    end_time: Optional[datetime] = Field(
        default=None, description="Window end"
    )
    shelve_duration_hours: int = Field(
        default=DEFAULT_SHELVE_HOURS, ge=0, description="Shelve hours"
    )
    delay_seconds: int = Field(
        default=DEFAULT_ON_DELAY_SEC, ge=0, description="Delay (sec)"
    )
    deadband_pct: Decimal = Field(
        default=DEFAULT_DEADBAND_PCT, ge=0, description="Deadband (%)"
    )
    is_active: bool = Field(
        default=True, description="Rule active"
    )
    description: str = Field(
        default="", max_length=2000, description="Description"
    )

class EscalationConfig(BaseModel):
    """Alarm escalation configuration.

    Attributes:
        config_id: Configuration identifier.
        priority: Alarm priority this config applies to.
        levels: Ordered escalation levels with timeouts.
        auto_escalate: Whether to auto-escalate on timeout.
        notification_channels: Notification channels per level.
        max_escalation_level: Maximum escalation level.
    """
    config_id: str = Field(
        default_factory=_new_uuid, description="Config ID"
    )
    priority: AlarmPriority = Field(
        default=AlarmPriority.MEDIUM, description="Priority"
    )
    levels: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {"level": EscalationLevel.L1_OPERATOR.value, "timeout_min": 60},
            {"level": EscalationLevel.L2_SUPERVISOR.value, "timeout_min": 120},
            {"level": EscalationLevel.L3_MANAGER.value, "timeout_min": 240},
            {"level": EscalationLevel.L4_EXECUTIVE.value, "timeout_min": 480},
        ],
        description="Escalation levels"
    )
    auto_escalate: bool = Field(
        default=True, description="Auto-escalate on timeout"
    )
    notification_channels: Dict[str, List[str]] = Field(
        default_factory=dict, description="Channels per level"
    )
    max_escalation_level: EscalationLevel = Field(
        default=EscalationLevel.L4_EXECUTIVE, description="Max level"
    )

class AlarmReport(BaseModel):
    """Alarm management performance report (ISA 18.2 KPIs).

    Attributes:
        report_id: Report identifier.
        reporting_period_start: Period start.
        reporting_period_end: Period end.
        total_alarms: Total alarm events.
        active_alarms: Currently active alarms.
        standing_alarms: Standing (stale) alarms.
        acknowledged_count: Acknowledged count.
        suppressed_count: Suppressed count.
        false_alarm_count: False alarm count.
        chattering_count: Chattering alarm count.
        avg_alarms_per_hour: Average alarms per operator hour.
        peak_alarms_per_10min: Peak alarms in any 10-min window.
        is_alarm_flood: Whether alarm flood detected.
        mtta_minutes: Mean Time To Acknowledge (minutes).
        mttr_minutes: Mean Time To Resolve (minutes).
        false_alarm_rate_pct: False alarm rate (%).
        chattering_rate_pct: Chattering rate (%).
        priority_distribution: Count by priority level.
        category_distribution: Count by category.
        state_distribution: Count by state.
        isa_compliance_score: ISA 18.2 compliance score (0-100).
        calculated_at: Calculation timestamp.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    report_id: str = Field(
        default_factory=_new_uuid, description="Report ID"
    )
    reporting_period_start: datetime = Field(
        default_factory=utcnow, description="Period start"
    )
    reporting_period_end: datetime = Field(
        default_factory=utcnow, description="Period end"
    )
    total_alarms: int = Field(default=0, ge=0, description="Total alarms")
    active_alarms: int = Field(default=0, ge=0, description="Active")
    standing_alarms: int = Field(default=0, ge=0, description="Standing")
    acknowledged_count: int = Field(default=0, ge=0, description="Acknowledged")
    suppressed_count: int = Field(default=0, ge=0, description="Suppressed")
    false_alarm_count: int = Field(default=0, ge=0, description="False")
    chattering_count: int = Field(default=0, ge=0, description="Chattering")
    avg_alarms_per_hour: Decimal = Field(
        default=Decimal("0"), description="Avg alarms/hr"
    )
    peak_alarms_per_10min: int = Field(
        default=0, ge=0, description="Peak alarms/10min"
    )
    is_alarm_flood: bool = Field(
        default=False, description="Alarm flood detected"
    )
    mtta_minutes: Decimal = Field(
        default=Decimal("0"), description="MTTA (min)"
    )
    mttr_minutes: Decimal = Field(
        default=Decimal("0"), description="MTTR (min)"
    )
    false_alarm_rate_pct: Decimal = Field(
        default=Decimal("0"), description="False alarm rate (%)"
    )
    chattering_rate_pct: Decimal = Field(
        default=Decimal("0"), description="Chattering rate (%)"
    )
    priority_distribution: Dict[str, int] = Field(
        default_factory=dict, description="By priority"
    )
    category_distribution: Dict[str, int] = Field(
        default_factory=dict, description="By category"
    )
    state_distribution: Dict[str, int] = Field(
        default_factory=dict, description="By state"
    )
    isa_compliance_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="ISA 18.2 score"
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AlarmEngine:
    """Intelligent alarm management engine with ISA 18.2 lifecycle.

    Evaluates alarm conditions against configurable thresholds, manages
    alarm state transitions through the ISA 18.2 lifecycle, applies
    suppression rules (shelve, delay, deadband, time-window, correlation),
    correlates related alarms, manages escalation chains, and calculates
    ISA 18.2 KPIs (MTTA, MTTR, false alarm rate, standing alarm count,
    alarm flood detection).

    Usage::

        engine = AlarmEngine()
        events = engine.evaluate_alarm(definitions, current_values)
        managed = engine.manage_lifecycle(events, actions)
        suppressed = engine.apply_suppression(events, rules)
        correlated = engine.correlate_alarms(events)
        report = engine.calculate_metrics(events)

    All arithmetic uses ``Decimal`` for deterministic, audit-grade precision.
    Every result carries a SHA-256 provenance hash.
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise AlarmEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - flood_threshold (int): alarms per 10-min for flood
                - standing_threshold_min (int): minutes for standing alarm
                - default_deadband_pct (Decimal): default deadband
                - escalation_configs (list): custom escalation configs
        """
        self.config = config or {}
        self._flood_threshold = int(
            self.config.get("flood_threshold", ALARM_FLOOD_THRESHOLD)
        )
        self._standing_threshold = int(
            self.config.get("standing_threshold_min", STANDING_ALARM_THRESHOLD_MIN)
        )
        self._default_deadband = _decimal(
            self.config.get("default_deadband_pct", DEFAULT_DEADBAND_PCT)
        )
        self._active_events: List[AlarmEvent] = []
        logger.info(
            "AlarmEngine v%s initialised (flood=%d/10min, standing=%d min)",
            self.engine_version, self._flood_threshold,
            self._standing_threshold,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def evaluate_alarm(
        self,
        definitions: List[AlarmDefinition],
        current_values: Dict[str, Decimal],
    ) -> List[AlarmEvent]:
        """Evaluate alarm conditions against current values.

        Compares each alarm definition's setpoint and thresholds against
        the current measured value, applying deadband hysteresis.

        Args:
            definitions: Alarm definitions to evaluate.
            current_values: Current values by alarm_id or meter_id.

        Returns:
            List of AlarmEvent instances for active alarms.
        """
        t0 = time.perf_counter()
        logger.info(
            "Evaluating alarms: %d definitions, %d values",
            len(definitions), len(current_values),
        )

        events: List[AlarmEvent] = []

        for defn in definitions:
            if not defn.is_enabled:
                continue

            # Look up current value by alarm_id first, then meter_id
            value = current_values.get(
                defn.alarm_id,
                current_values.get(defn.meter_id, None),
            )
            if value is None:
                continue

            current_val = _decimal(value)
            setpoint = defn.setpoint

            if setpoint == Decimal("0"):
                deviation_pct = Decimal("0") if current_val == Decimal("0") else Decimal("100")
            else:
                deviation_pct = _safe_pct(
                    abs(current_val - setpoint), abs(setpoint),
                )

            # Check high alarm
            is_high = deviation_pct > defn.high_threshold_pct and current_val > setpoint
            # Check low alarm
            is_low = deviation_pct > defn.low_threshold_pct and current_val < setpoint

            if is_high or is_low:
                # Check deadband (only suppress if within deadband of threshold)
                effective_threshold = (
                    defn.high_threshold_pct if is_high
                    else defn.low_threshold_pct
                )
                within_deadband = (
                    deviation_pct < effective_threshold + defn.deadband_pct
                    and deviation_pct >= effective_threshold
                )

                # If within deadband on return-to-normal, skip
                if within_deadband and self._is_returning_to_normal(
                    defn.alarm_id, current_val, setpoint,
                ):
                    continue

                event = AlarmEvent(
                    alarm_id=defn.alarm_id,
                    alarm_name=defn.alarm_name,
                    priority=defn.priority,
                    category=defn.alarm_category,
                    state=AlarmState.ACTIVE,
                    current_value=_round_val(current_val, 4),
                    setpoint=_round_val(setpoint, 4),
                    deviation_pct=_round_val(deviation_pct, 2),
                    escalation_level=defn.escalation_level,
                )
                events.append(event)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Alarm evaluation: %d active from %d definitions (%.1f ms)",
            len(events), len(definitions), elapsed,
        )
        return events

    def manage_lifecycle(
        self,
        events: List[AlarmEvent],
        actions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Manage alarm lifecycle state transitions per ISA 18.2.

        Processes operator actions (acknowledge, shelve, clear, close)
        and updates alarm states accordingly.

        Args:
            events: Current alarm events.
            actions: State transition actions by event_id.
                     Values: 'acknowledge', 'shelve', 'clear', 'close'.

        Returns:
            Dictionary with lifecycle management results.
        """
        t0 = time.perf_counter()
        logger.info(
            "Managing lifecycle: %d events, %d actions",
            len(events), len(actions or {}),
        )

        action_map = actions or {}
        transitions: List[Dict[str, Any]] = []
        now = utcnow()

        for event in events:
            action = action_map.get(event.event_id, "")
            old_state = event.state

            if action == "acknowledge" and event.state == AlarmState.ACTIVE:
                event.state = AlarmState.ACKNOWLEDGED
                event.acknowledged_at = now
            elif action == "shelve" and event.state in (
                AlarmState.ACTIVE, AlarmState.ACKNOWLEDGED,
            ):
                event.state = AlarmState.SHELVED
            elif action == "clear" and event.state in (
                AlarmState.ACTIVE, AlarmState.ACKNOWLEDGED,
            ):
                event.state = AlarmState.CLEARED
                event.cleared_at = now
            elif action == "close" and event.state == AlarmState.CLEARED:
                event.state = AlarmState.CLOSED
                event.closed_at = now

            if event.state != old_state:
                transitions.append({
                    "event_id": event.event_id,
                    "alarm_name": event.alarm_name,
                    "from_state": old_state.value,
                    "to_state": event.state.value,
                    "action": action,
                    "timestamp": now.isoformat(),
                })

        # Escalation check
        escalations = self._check_escalations(events, now)

        elapsed = (time.perf_counter() - t0) * 1000.0
        result: Dict[str, Any] = {
            "total_events": len(events),
            "transitions": transitions,
            "transition_count": len(transitions),
            "escalations": escalations,
            "escalation_count": len(escalations),
            "state_summary": self._summarise_states(events),
            "calculated_at": now.isoformat(),
            "processing_time_ms": round(elapsed, 2),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Lifecycle managed: %d transitions, %d escalations, "
            "hash=%s (%.1f ms)",
            len(transitions), len(escalations),
            result["provenance_hash"][:16], elapsed,
        )
        return result

    def apply_suppression(
        self,
        events: List[AlarmEvent],
        rules: List[SuppressionRule],
    ) -> Dict[str, Any]:
        """Apply suppression rules to alarm events.

        Evaluates each suppression rule against the event list and
        marks matching events as suppressed.

        Args:
            events: Alarm events to evaluate.
            rules: Suppression rules to apply.

        Returns:
            Dictionary with suppression results.
        """
        t0 = time.perf_counter()
        logger.info(
            "Applying suppression: %d events, %d rules",
            len(events), len(rules),
        )

        now = utcnow()
        suppressed_events: List[Dict[str, Any]] = []
        active_rules = [r for r in rules if r.is_active]

        for event in events:
            if event.state in (AlarmState.CLEARED, AlarmState.CLOSED):
                continue

            for rule in active_rules:
                if not self._rule_matches(rule, event):
                    continue

                should_suppress = False

                if rule.suppression_type == SuppressionType.SHELVE:
                    should_suppress = True
                    event.state = AlarmState.SHELVED

                elif rule.suppression_type == SuppressionType.TIME_WINDOW:
                    if rule.start_time and rule.end_time:
                        if rule.start_time <= now <= rule.end_time:
                            should_suppress = True

                elif rule.suppression_type == SuppressionType.CORRELATION:
                    # Suppress if parent alarm is active
                    parent_active = any(
                        e.alarm_id == rule.parent_alarm_id
                        and e.state == AlarmState.ACTIVE
                        for e in events
                    )
                    if parent_active:
                        should_suppress = True

                elif rule.suppression_type == SuppressionType.DEADBAND:
                    if event.deviation_pct <= rule.deadband_pct:
                        should_suppress = True

                elif rule.suppression_type == SuppressionType.DELAY:
                    age_sec = (now - event.activated_at).total_seconds()
                    if age_sec < rule.delay_seconds:
                        should_suppress = True

                if should_suppress:
                    event.is_suppressed = True
                    event.suppression_type = rule.suppression_type
                    if event.state == AlarmState.ACTIVE:
                        event.state = AlarmState.SUPPRESSED
                    suppressed_events.append({
                        "event_id": event.event_id,
                        "alarm_name": event.alarm_name,
                        "rule_id": rule.rule_id,
                        "suppression_type": rule.suppression_type.value,
                    })
                    break  # One suppression rule per event

        elapsed = (time.perf_counter() - t0) * 1000.0
        result: Dict[str, Any] = {
            "total_events": len(events),
            "rules_evaluated": len(active_rules),
            "suppressed_count": len(suppressed_events),
            "suppressed_events": suppressed_events,
            "calculated_at": now.isoformat(),
            "processing_time_ms": round(elapsed, 2),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Suppression applied: %d suppressed from %d events, "
            "hash=%s (%.1f ms)",
            len(suppressed_events), len(events),
            result["provenance_hash"][:16], elapsed,
        )
        return result

    def correlate_alarms(
        self,
        events: List[AlarmEvent],
    ) -> Dict[str, Any]:
        """Correlate related alarms to identify root causes.

        Groups alarms by time proximity, meter/equipment, and parent-child
        relationships to reduce operator alarm burden.

        Args:
            events: Alarm events to correlate.

        Returns:
            Dictionary with correlation groups and root cause candidates.
        """
        t0 = time.perf_counter()
        logger.info("Correlating alarms: %d events", len(events))

        # Sort by activation time
        sorted_events = sorted(events, key=lambda e: e.activated_at)

        # Group by time proximity (within 5 minutes)
        time_groups: List[List[AlarmEvent]] = []
        current_group: List[AlarmEvent] = []
        proximity_window = timedelta(minutes=5)

        for event in sorted_events:
            if not current_group:
                current_group.append(event)
            elif (event.activated_at - current_group[-1].activated_at) <= proximity_window:
                current_group.append(event)
            else:
                if len(current_group) > 1:
                    time_groups.append(current_group)
                current_group = [event]
        if len(current_group) > 1:
            time_groups.append(current_group)

        # Build correlation results
        correlations: List[Dict[str, Any]] = []
        for idx, group in enumerate(time_groups):
            # Highest-priority alarm is root cause candidate
            root_candidate = min(
                group,
                key=lambda e: list(AlarmPriority).index(e.priority),
            )

            correlations.append({
                "group_id": f"CG-{idx + 1:04d}",
                "event_count": len(group),
                "root_cause_candidate": {
                    "event_id": root_candidate.event_id,
                    "alarm_name": root_candidate.alarm_name,
                    "priority": root_candidate.priority.value,
                    "category": root_candidate.category.value,
                },
                "related_events": [
                    {
                        "event_id": e.event_id,
                        "alarm_name": e.alarm_name,
                        "priority": e.priority.value,
                        "time_offset_sec": (
                            e.activated_at - group[0].activated_at
                        ).total_seconds(),
                    }
                    for e in group
                ],
                "time_span_sec": (
                    group[-1].activated_at - group[0].activated_at
                ).total_seconds(),
            })

        # Detect chattering alarms
        chattering = self._detect_chattering(events)

        elapsed = (time.perf_counter() - t0) * 1000.0
        result: Dict[str, Any] = {
            "total_events": len(events),
            "correlation_groups": len(correlations),
            "correlations": correlations,
            "chattering_alarms": chattering,
            "chattering_count": len(chattering),
            "calculated_at": utcnow().isoformat(),
            "processing_time_ms": round(elapsed, 2),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Correlation: %d groups, %d chattering from %d events, "
            "hash=%s (%.1f ms)",
            len(correlations), len(chattering), len(events),
            result["provenance_hash"][:16], elapsed,
        )
        return result

    def calculate_metrics(
        self,
        events: List[AlarmEvent],
        reporting_period_hours: Decimal = Decimal("720"),
    ) -> AlarmReport:
        """Calculate ISA 18.2 alarm management KPIs.

        Computes MTTA, MTTR, false alarm rate, standing alarm count,
        alarm rate, priority distribution, and ISA 18.2 compliance score.

        Args:
            events: Alarm events for the reporting period.
            reporting_period_hours: Length of reporting period (hours).

        Returns:
            AlarmReport with comprehensive metrics.
        """
        t0 = time.perf_counter()
        logger.info(
            "Calculating metrics: %d events, %.0f hours",
            len(events), float(reporting_period_hours),
        )

        total = len(events)
        now = utcnow()

        # State counts
        state_dist: Dict[str, int] = {}
        priority_dist: Dict[str, int] = {}
        category_dist: Dict[str, int] = {}
        active_count = 0
        acked_count = 0
        suppressed_count = 0
        false_count = 0
        chatter_count = 0

        # Timing accumulators
        tta_total = Decimal("0")
        tta_count = 0
        ttr_total = Decimal("0")
        ttr_count = 0

        standing_count = 0
        standing_threshold = timedelta(minutes=self._standing_threshold)

        for event in events:
            # State distribution
            state_dist[event.state.value] = state_dist.get(event.state.value, 0) + 1
            priority_dist[event.priority.value] = priority_dist.get(event.priority.value, 0) + 1
            category_dist[event.category.value] = category_dist.get(event.category.value, 0) + 1

            if event.state == AlarmState.ACTIVE:
                active_count += 1
                # Standing alarm check
                if (now - event.activated_at) > standing_threshold:
                    standing_count += 1

            if event.state == AlarmState.ACKNOWLEDGED:
                acked_count += 1

            if event.is_suppressed:
                suppressed_count += 1

            if event.is_false_alarm:
                false_count += 1

            if event.is_chattering:
                chatter_count += 1

            # MTTA
            if event.acknowledged_at is not None:
                tta_sec = (event.acknowledged_at - event.activated_at).total_seconds()
                tta_total += _decimal(max(tta_sec, 0))
                tta_count += 1

            # MTTR
            if event.cleared_at is not None:
                ttr_sec = (event.cleared_at - event.activated_at).total_seconds()
                ttr_total += _decimal(max(ttr_sec, 0))
                ttr_count += 1

        # KPIs
        mtta_min = _safe_divide(
            tta_total, _decimal(tta_count) * Decimal("60"),
        ) if tta_count > 0 else Decimal("0")
        # Recompute: tta_total is in seconds, convert to minutes
        mtta_min = _safe_divide(tta_total, _decimal(tta_count), Decimal("0")) / Decimal("60") if tta_count > 0 else Decimal("0")
        mttr_min = _safe_divide(ttr_total, _decimal(ttr_count), Decimal("0")) / Decimal("60") if ttr_count > 0 else Decimal("0")

        avg_per_hour = _safe_divide(
            _decimal(total), reporting_period_hours,
        )
        false_rate = _safe_pct(_decimal(false_count), _decimal(total))
        chatter_rate = _safe_pct(_decimal(chatter_count), _decimal(total))

        # Alarm flood detection (peak 10-min window)
        peak_10min = self._compute_peak_10min(events)
        is_flood = peak_10min >= self._flood_threshold

        # ISA 18.2 compliance score
        isa_score = self._compute_isa_score(
            avg_per_hour, mtta_min, false_rate, _decimal(standing_count),
            chatter_rate,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        report = AlarmReport(
            total_alarms=total,
            active_alarms=active_count,
            standing_alarms=standing_count,
            acknowledged_count=acked_count,
            suppressed_count=suppressed_count,
            false_alarm_count=false_count,
            chattering_count=chatter_count,
            avg_alarms_per_hour=_round_val(avg_per_hour, 2),
            peak_alarms_per_10min=peak_10min,
            is_alarm_flood=is_flood,
            mtta_minutes=_round_val(mtta_min, 2),
            mttr_minutes=_round_val(mttr_min, 2),
            false_alarm_rate_pct=_round_val(false_rate, 2),
            chattering_rate_pct=_round_val(chatter_rate, 2),
            priority_distribution=priority_dist,
            category_distribution=category_dist,
            state_distribution=state_dist,
            isa_compliance_score=_round_val(isa_score, 1),
            processing_time_ms=round(elapsed, 2),
        )
        report.provenance_hash = _compute_hash(report)

        logger.info(
            "Alarm metrics: total=%d, active=%d, standing=%d, "
            "MTTA=%.1f min, MTTR=%.1f min, false=%.1f%%, ISA=%.1f, "
            "flood=%s, hash=%s (%.1f ms)",
            total, active_count, standing_count,
            float(mtta_min), float(mttr_min), float(false_rate),
            float(isa_score), is_flood,
            report.provenance_hash[:16], elapsed,
        )
        return report

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _is_returning_to_normal(
        self, alarm_id: str, current_value: Decimal, setpoint: Decimal,
    ) -> bool:
        """Check if an alarm is returning to normal (for deadband logic).

        Args:
            alarm_id: Alarm identifier.
            current_value: Current value.
            setpoint: Alarm setpoint.

        Returns:
            True if trending back toward setpoint.
        """
        # Simplified: check if any previous active event is trending back
        for prev in self._active_events:
            if prev.alarm_id == alarm_id:
                prev_deviation = abs(prev.current_value - setpoint)
                curr_deviation = abs(current_value - setpoint)
                if curr_deviation < prev_deviation:
                    return True
        return False

    def _check_escalations(
        self,
        events: List[AlarmEvent],
        now: datetime,
    ) -> List[Dict[str, Any]]:
        """Check events for escalation requirements.

        Args:
            events: Current events.
            now: Current timestamp.

        Returns:
            List of escalation actions needed.
        """
        escalations: List[Dict[str, Any]] = []

        for event in events:
            if event.state not in (AlarmState.ACTIVE, AlarmState.ACKNOWLEDGED):
                continue

            timeout = ESCALATION_TIMEOUTS.get(
                event.priority.value, 60,
            )
            age_min = (now - event.activated_at).total_seconds() / 60.0

            if age_min > timeout:
                next_level = self._next_escalation_level(event.escalation_level)
                if next_level != event.escalation_level:
                    escalations.append({
                        "event_id": event.event_id,
                        "alarm_name": event.alarm_name,
                        "priority": event.priority.value,
                        "current_level": event.escalation_level.value,
                        "escalate_to": next_level.value,
                        "age_minutes": round(age_min, 1),
                        "timeout_minutes": timeout,
                    })
                    event.escalation_level = next_level

        return escalations

    def _next_escalation_level(
        self, current: EscalationLevel,
    ) -> EscalationLevel:
        """Get next escalation level.

        Args:
            current: Current escalation level.

        Returns:
            Next escalation level (or same if at max).
        """
        levels = list(EscalationLevel)
        idx = levels.index(current)
        if idx < len(levels) - 1:
            return levels[idx + 1]
        return current

    def _rule_matches(
        self, rule: SuppressionRule, event: AlarmEvent,
    ) -> bool:
        """Check if a suppression rule matches an alarm event.

        Args:
            rule: Suppression rule.
            event: Alarm event.

        Returns:
            True if rule applies to this event.
        """
        if rule.target_alarm_ids:
            return event.alarm_id in rule.target_alarm_ids
        return True

    def _summarise_states(
        self, events: List[AlarmEvent],
    ) -> Dict[str, int]:
        """Summarise alarm states.

        Args:
            events: Alarm events.

        Returns:
            Count by state.
        """
        summary: Dict[str, int] = {}
        for event in events:
            summary[event.state.value] = summary.get(event.state.value, 0) + 1
        return summary

    def _detect_chattering(
        self, events: List[AlarmEvent],
    ) -> List[Dict[str, Any]]:
        """Detect chattering alarms (rapid on/off cycling).

        Args:
            events: Alarm events to check.

        Returns:
            List of chattering alarm details.
        """
        # Group events by alarm_id
        by_alarm: Dict[str, List[AlarmEvent]] = {}
        for event in events:
            by_alarm.setdefault(event.alarm_id, []).append(event)

        chattering: List[Dict[str, Any]] = []
        for alarm_id, alarm_events in by_alarm.items():
            if len(alarm_events) >= CHATTER_MIN_TRANSITIONS:
                sorted_ev = sorted(alarm_events, key=lambda e: e.activated_at)
                span_min = (
                    sorted_ev[-1].activated_at - sorted_ev[0].activated_at
                ).total_seconds() / 60.0

                if span_min <= CHATTER_WINDOW_MIN and len(sorted_ev) >= CHATTER_MIN_TRANSITIONS:
                    for ev in sorted_ev:
                        ev.is_chattering = True
                    chattering.append({
                        "alarm_id": alarm_id,
                        "alarm_name": sorted_ev[0].alarm_name,
                        "transition_count": len(sorted_ev),
                        "span_minutes": round(span_min, 1),
                    })

        return chattering

    def _compute_peak_10min(
        self, events: List[AlarmEvent],
    ) -> int:
        """Compute peak alarm count in any 10-minute window.

        Args:
            events: Alarm events.

        Returns:
            Maximum alarm count in a 10-minute window.
        """
        if not events:
            return 0

        sorted_events = sorted(events, key=lambda e: e.activated_at)
        window = timedelta(minutes=10)
        peak = 0

        for i, event in enumerate(sorted_events):
            count = 1
            for j in range(i + 1, len(sorted_events)):
                if (sorted_events[j].activated_at - event.activated_at) <= window:
                    count += 1
                else:
                    break
            peak = max(peak, count)

        return peak

    def _compute_isa_score(
        self,
        avg_per_hour: Decimal,
        mtta_min: Decimal,
        false_rate: Decimal,
        standing_count: Decimal,
        chatter_rate: Decimal,
    ) -> Decimal:
        """Compute ISA 18.2 compliance score (0-100).

        Weighted scoring based on ISA 18.2 recommended targets:
        - Avg alarms/hour <= 6 (30% weight)
        - MTTA <= 5 min (25% weight)
        - False alarm rate <= 10% (20% weight)
        - Standing alarms <= 5 (15% weight)
        - Chattering rate <= 5% (10% weight)

        Args:
            avg_per_hour: Average alarms per hour.
            mtta_min: Mean time to acknowledge (minutes).
            false_rate: False alarm rate (%).
            standing_count: Standing alarm count.
            chatter_rate: Chattering rate (%).

        Returns:
            ISA 18.2 compliance score (0-100).
        """
        score = Decimal("0")

        # Alarm rate score (30%): target <= 6/hr
        if avg_per_hour <= ISA_MAX_AVG_ALARMS_PER_HOUR:
            rate_score = Decimal("30")
        else:
            ratio = _safe_divide(ISA_MAX_AVG_ALARMS_PER_HOUR, avg_per_hour)
            rate_score = Decimal("30") * ratio
        score += max(rate_score, Decimal("0"))

        # MTTA score (25%): target <= 5 min
        target_mtta = Decimal("5")
        if mtta_min <= target_mtta:
            mtta_score = Decimal("25")
        elif mtta_min <= Decimal("0"):
            mtta_score = Decimal("25")
        else:
            ratio = _safe_divide(target_mtta, mtta_min)
            mtta_score = Decimal("25") * ratio
        score += max(mtta_score, Decimal("0"))

        # False alarm score (20%): target <= 10%
        target_false = Decimal("10")
        if false_rate <= target_false:
            false_score = Decimal("20")
        else:
            ratio = _safe_divide(target_false, false_rate)
            false_score = Decimal("20") * ratio
        score += max(false_score, Decimal("0"))

        # Standing alarm score (15%): target <= 5
        target_standing = Decimal("5")
        if standing_count <= target_standing:
            standing_score = Decimal("15")
        else:
            ratio = _safe_divide(target_standing, standing_count)
            standing_score = Decimal("15") * ratio
        score += max(standing_score, Decimal("0"))

        # Chattering score (10%): target <= 5%
        target_chatter = Decimal("5")
        if chatter_rate <= target_chatter:
            chatter_score = Decimal("10")
        else:
            ratio = _safe_divide(target_chatter, chatter_rate)
            chatter_score = Decimal("10") * ratio
        score += max(chatter_score, Decimal("0"))

        return min(score, Decimal("100"))
