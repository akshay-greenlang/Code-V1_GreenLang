# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Emergency Shutdown Module

Emergency shutdown handler for insulation monitoring systems. Manages detection
and response to critical insulation failures, burn hazards, thermal runaway
conditions, and system integrity failures.

CRITICAL: This module provides RECOMMENDATIONS ONLY. GL-015 does NOT directly
actuate any shutdown procedures. All shutdown actions are performed by
independent plant safety systems or manual operator intervention.

Emergency Conditions Handled:
1. Surface temperature exceeding burn risk threshold (>55C per ASTM C1055)
2. Catastrophic insulation failure detected
3. Multiple concurrent hot spots
4. Thermal runaway conditions (rapid temperature rise)
5. Critical system integrity failures
6. Out-of-distribution sensor readings

Shutdown Levels:
- Level 1 (ALERT): Generate warnings, continue monitoring
- Level 2 (PROTECT): Enable protective measures, notify operators
- Level 3 (LIMIT): Reduce recommendations, escalate to engineering
- Level 4 (SHUTDOWN): Full recommendation suspension, require manual override

Safety Principles:
- Recommendations only, no autonomous control
- Deterministic logic only (no ML for safety decisions)
- Full SHA-256 provenance tracking
- Immutable audit log of all events
- Default to safe state
- Thread-safe implementation

Standards References:
- ASTM C1055: Standard Guide for Heated System Surface Conditions
- OSHA 29 CFR 1910.132: Personal Protective Equipment
- OSHA 29 CFR 1910.147: Control of Hazardous Energy
- IEC 61511: Safety Instrumented Systems
- NFPA 86: Standard for Ovens and Furnaces

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, IntEnum
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator

from .exceptions import (
    InsulscanSafetyError,
    ViolationContext,
    ViolationDetails,
    ViolationSeverity,
    SafetyDomain,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class ShutdownLevel(IntEnum):
    """
    Emergency shutdown levels for insulation monitoring systems.

    Levels follow escalating severity with corresponding actions:
    - ALERT (1): Warning only, continue normal operation
    - PROTECT (2): Protective measures, operator notification
    - LIMIT (3): Reduce recommendations, engineering escalation
    - SHUTDOWN (4): Suspend recommendations, require manual override
    """

    ALERT = 1
    PROTECT = 2
    LIMIT = 3
    SHUTDOWN = 4

    def get_description(self) -> str:
        """Get human-readable description of the shutdown level."""
        descriptions = {
            ShutdownLevel.ALERT: "Generate warnings, continue monitoring",
            ShutdownLevel.PROTECT: "Enable protective measures, notify operators",
            ShutdownLevel.LIMIT: "Reduce recommendations, escalate to engineering",
            ShutdownLevel.SHUTDOWN: "Suspend recommendations, require manual override",
        }
        return descriptions.get(self, "Unknown level")


class ConditionType(str, Enum):
    """Types of emergency conditions for insulation systems."""

    BURN_RISK = "burn_risk"
    INSULATION_FAILURE = "insulation_failure"
    MULTIPLE_HOT_SPOTS = "multiple_hot_spots"
    THERMAL_RUNAWAY = "thermal_runaway"
    SYSTEM_INTEGRITY = "system_integrity"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    SENSOR_ANOMALY = "sensor_anomaly"
    COMMUNICATION_LOSS = "communication_loss"
    HIGH_HEAT_LOSS = "high_heat_loss"
    RAPID_DEGRADATION = "rapid_degradation"


class ConditionState(str, Enum):
    """Current state of a shutdown condition."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class EscalationLevel(str, Enum):
    """Escalation hierarchy for emergency conditions."""

    NONE = "none"
    OPERATOR = "operator"
    SUPERVISOR = "supervisor"
    ENGINEER = "engineer"
    SAFETY_TEAM = "safety_team"
    PLANT_MANAGER = "plant_manager"


class ResponseAction(str, Enum):
    """Recommended response actions for emergency conditions."""

    CONTINUE_MONITORING = "continue_monitoring"
    INCREASE_MONITORING = "increase_monitoring"
    NOTIFY_OPERATOR = "notify_operator"
    ESCALATE_ENGINEERING = "escalate_engineering"
    RESTRICT_ACCESS = "restrict_access"
    TEMPORARY_BARRIER = "temporary_barrier"
    EMERGENCY_REPAIR = "emergency_repair"
    PROCESS_REDUCTION = "process_reduction"
    ISOLATE_AREA = "isolate_area"
    SUSPEND_RECOMMENDATIONS = "suspend_recommendations"


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class ThermalMeasurement:
    """
    Thermal measurement data for insulation assessment.

    Attributes:
        asset_id: Insulation asset identifier
        surface_temp_c: Surface temperature in Celsius
        ambient_temp_c: Ambient temperature in Celsius
        process_temp_c: Process fluid temperature in Celsius
        timestamp: Measurement timestamp
        location: Physical location
        sensor_id: Sensor identifier
        heat_loss_w_m2: Calculated heat loss W/m2
        additional_data: Additional measurement data
    """

    asset_id: str
    surface_temp_c: float
    ambient_temp_c: float
    process_temp_c: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    location: str = ""
    sensor_id: str = ""
    heat_loss_w_m2: float = 0.0
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShutdownCondition:
    """
    Detected emergency shutdown condition.

    Immutable record of a detected condition with full provenance tracking.

    Attributes:
        condition_id: Unique condition identifier
        condition_type: Type of emergency condition
        severity: Shutdown level (1-4)
        description: Human-readable description
        detected_at: When condition was detected
        asset_id: Affected asset identifier (optional)
        measurement_values: Relevant measurement data
        threshold_violated: The threshold that was exceeded
        recommended_actions: List of recommended response actions
        standard_reference: Applicable safety standard
        provenance_hash: SHA-256 hash for audit trail
    """

    condition_id: str
    condition_type: ConditionType
    severity: ShutdownLevel
    description: str
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    asset_id: Optional[str] = None
    measurement_values: Dict[str, float] = field(default_factory=dict)
    threshold_violated: str = ""
    recommended_actions: List[ResponseAction] = field(default_factory=list)
    standard_reference: str = ""
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate SHA-256 provenance hash for audit trail."""
        if not self.provenance_hash:
            content = (
                f"{self.condition_id}|{self.condition_type.value}|"
                f"{self.severity.value}|{self.detected_at.isoformat()}|"
                f"{self.asset_id or 'none'}|"
                f"{sorted(self.measurement_values.items())}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "condition_id": self.condition_id,
            "condition_type": self.condition_type.value,
            "severity": self.severity.value,
            "severity_name": self.severity.name,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "asset_id": self.asset_id,
            "measurement_values": self.measurement_values,
            "threshold_violated": self.threshold_violated,
            "recommended_actions": [a.value for a in self.recommended_actions],
            "standard_reference": self.standard_reference,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class ShutdownEvent:
    """
    Record of a shutdown event with full lifecycle tracking.

    Attributes:
        event_id: Unique event identifier
        condition: The triggering condition
        state: Current event state
        escalation_level: Current escalation level
        created_at: When event was created
        acknowledged_at: When event was acknowledged
        acknowledged_by: Who acknowledged the event
        resolved_at: When event was resolved
        resolved_by: Who resolved the event
        resolution_notes: Resolution details
        escalation_history: History of escalations
        actions_taken: Actions taken in response
        provenance_hash: SHA-256 hash for audit trail
    """

    event_id: str
    condition: ShutdownCondition
    state: ConditionState = ConditionState.ACTIVE
    escalation_level: EscalationLevel = EscalationLevel.NONE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: str = ""
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate SHA-256 provenance hash for audit trail."""
        if not self.provenance_hash:
            content = (
                f"{self.event_id}|{self.condition.condition_id}|"
                f"{self.created_at.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()

    @property
    def age_minutes(self) -> float:
        """Get event age in minutes."""
        now = datetime.now(timezone.utc)
        return (now - self.created_at).total_seconds() / 60

    @property
    def is_active(self) -> bool:
        """Check if event is still active."""
        return self.state in (ConditionState.ACTIVE, ConditionState.ESCALATED)


@dataclass
class ShutdownResult:
    """
    Result of triggering a shutdown response.

    Attributes:
        success: Whether shutdown was triggered successfully
        shutdown_level: The activated shutdown level
        events_created: List of events created
        notifications_sent: Notification targets
        recommendations_suspended: Whether recommendations are suspended
        manual_override_required: Whether manual override is needed
        timestamp: When shutdown was triggered
        provenance_hash: SHA-256 hash for audit trail
    """

    success: bool
    shutdown_level: ShutdownLevel
    events_created: List[str] = field(default_factory=list)
    notifications_sent: List[str] = field(default_factory=list)
    recommendations_suspended: bool = False
    manual_override_required: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate SHA-256 provenance hash for audit trail."""
        if not self.provenance_hash:
            content = (
                f"{self.success}|{self.shutdown_level.value}|"
                f"{self.timestamp.isoformat()}|{len(self.events_created)}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# CONFIGURATION
# =============================================================================


class EmergencyShutdownConfig(BaseModel):
    """
    Configuration for insulation emergency shutdown handler.

    Default values follow ASTM C1055 and OSHA guidelines for personnel safety.

    Attributes:
        burn_temp_threshold_c: Surface temperature for burn risk (ASTM C1055)
        burn_temp_fahrenheit: Equivalent in Fahrenheit
        hot_spot_count_threshold: Number of hot spots for multiple hot spot alert
        thermal_runaway_rate_c_per_min: Temperature rise rate for thermal runaway
        high_heat_loss_threshold_w_m2: Heat loss threshold for alert
        auto_shutdown_enabled: Enable automatic shutdown actions (recommendations only)
        notification_endpoints: List of notification targets
        cooldown_period_minutes: Cooldown after condition is resolved
        max_active_conditions: Maximum active conditions before system alert
        escalation_timeout_minutes: Time before auto-escalation
        sensor_deviation_threshold: OOD sensor deviation threshold (std deviations)
        rapid_degradation_rate_per_day: Condition score change rate for alert
    """

    # Temperature thresholds per ASTM C1055
    burn_temp_threshold_c: float = Field(
        default=55.0,
        ge=40.0,
        le=80.0,
        description="Surface temperature threshold for burn risk (ASTM C1055: 55C/131F)"
    )

    @property
    def burn_temp_fahrenheit(self) -> float:
        """Get burn temperature threshold in Fahrenheit."""
        return self.burn_temp_threshold_c * 9/5 + 32

    # Hot spot thresholds
    hot_spot_count_threshold: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of concurrent hot spots to trigger multiple hot spot alert"
    )

    hot_spot_proximity_meters: float = Field(
        default=5.0,
        ge=0.5,
        le=50.0,
        description="Maximum distance between hot spots to consider related"
    )

    # Thermal runaway detection
    thermal_runaway_rate_c_per_min: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Temperature rise rate (C/min) indicating thermal runaway"
    )

    thermal_runaway_window_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Time window for thermal runaway detection"
    )

    # Heat loss thresholds
    high_heat_loss_threshold_w_m2: float = Field(
        default=1000.0,
        ge=100.0,
        le=5000.0,
        description="Heat loss threshold for high heat loss alert (W/m2)"
    )

    critical_heat_loss_threshold_w_m2: float = Field(
        default=2000.0,
        ge=500.0,
        le=10000.0,
        description="Heat loss threshold for critical alert (W/m2)"
    )

    # System behavior
    auto_shutdown_enabled: bool = Field(
        default=False,
        description="Enable automatic shutdown (recommendations only by default)"
    )

    notification_endpoints: List[str] = Field(
        default_factory=list,
        description="List of notification endpoint URLs or identifiers"
    )

    cooldown_period_minutes: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Cooldown period after condition is resolved"
    )

    max_active_conditions: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Maximum active conditions before system-level alert"
    )

    # Escalation settings
    escalation_timeout_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Time before automatic escalation of unacknowledged conditions"
    )

    level2_escalation_minutes: int = Field(
        default=10,
        ge=2,
        le=30,
        description="Time before Level 2 conditions escalate"
    )

    level3_escalation_minutes: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Time before Level 3 conditions escalate"
    )

    level4_escalation_minutes: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Time before Level 4 conditions escalate"
    )

    # OOD detection
    sensor_deviation_threshold: float = Field(
        default=4.0,
        ge=2.0,
        le=10.0,
        description="Standard deviations for OOD sensor detection"
    )

    # Degradation detection
    rapid_degradation_rate_per_day: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Condition score change rate per day for rapid degradation alert"
    )

    # Audit settings
    retain_resolved_hours: int = Field(
        default=168,
        ge=24,
        le=720,
        description="Hours to retain resolved conditions (default: 7 days)"
    )

    audit_log_max_entries: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Maximum audit log entries to retain in memory"
    )

    @field_validator("burn_temp_threshold_c")
    @classmethod
    def validate_burn_threshold(cls, v: float) -> float:
        """Validate burn threshold is within ASTM C1055 guidance."""
        if v < 55.0:
            logger.warning(
                f"burn_temp_threshold_c={v}C is below ASTM C1055 recommendation of 55C. "
                f"This may generate excessive alerts."
            )
        if v > 60.0:
            logger.warning(
                f"burn_temp_threshold_c={v}C exceeds ASTM C1055 recommendation. "
                f"Personnel safety may be compromised."
            )
        return v


# =============================================================================
# EMERGENCY SHUTDOWN HANDLER
# =============================================================================


class InsulationEmergencyShutdown:
    """
    Emergency shutdown handler for insulation monitoring systems.

    This handler detects emergency conditions, triggers appropriate responses,
    and maintains an immutable audit trail. It follows fail-safe principles
    and provides recommendations only - no autonomous control actions.

    Safety Principles:
    - All decisions use deterministic logic (no ML)
    - Full SHA-256 provenance tracking
    - Immutable audit log of all events
    - Default to safe state on uncertainty
    - Recommendations only, no direct actuation

    Thread Safety:
    - All public methods are thread-safe via internal locking

    Example:
        >>> config = EmergencyShutdownConfig(burn_temp_threshold_c=55.0)
        >>> handler = InsulationEmergencyShutdown(config)
        >>>
        >>> # Evaluate thermal measurements
        >>> measurements = [
        ...     ThermalMeasurement(
        ...         asset_id="INS-001",
        ...         surface_temp_c=62.0,
        ...         ambient_temp_c=25.0,
        ...     )
        ... ]
        >>> conditions = handler.evaluate_conditions(measurements)
        >>>
        >>> # Trigger shutdown if conditions detected
        >>> if conditions:
        ...     result = handler.trigger_shutdown(conditions)
        ...     if result.manual_override_required:
        ...         print("Manual intervention required")

    Author: GL-BackendDeveloper
    Version: 1.0.0
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        config: Optional[EmergencyShutdownConfig] = None,
        notification_callback: Optional[Callable[[ShutdownEvent, ShutdownLevel], None]] = None,
        escalation_callback: Optional[Callable[[ShutdownEvent, EscalationLevel], None]] = None,
    ) -> None:
        """
        Initialize insulation emergency shutdown handler.

        Args:
            config: Emergency shutdown configuration
            notification_callback: Callback for sending notifications
            escalation_callback: Callback for escalation events
        """
        self.config = config or EmergencyShutdownConfig()
        self._lock = threading.RLock()

        # Active conditions and events
        self._active_events: Dict[str, ShutdownEvent] = {}
        self._resolved_events: Deque[ShutdownEvent] = deque(
            maxlen=self.config.audit_log_max_entries
        )

        # Condition tracking
        self._condition_history: Dict[str, List[ShutdownCondition]] = {}
        self._last_measurements: Dict[str, ThermalMeasurement] = {}
        self._temperature_history: Dict[str, Deque[Tuple[datetime, float]]] = {}

        # System state
        self._recommendations_suspended: bool = False
        self._manual_override_active: bool = False
        self._current_shutdown_level: ShutdownLevel = ShutdownLevel.ALERT

        # Callbacks
        self._notification_callback = notification_callback
        self._escalation_callback = escalation_callback

        # Audit log
        self._audit_log: Deque[Dict[str, Any]] = deque(
            maxlen=self.config.audit_log_max_entries
        )

        logger.info(
            f"InsulationEmergencyShutdown initialized: "
            f"burn_threshold={self.config.burn_temp_threshold_c}C, "
            f"auto_shutdown={self.config.auto_shutdown_enabled}"
        )

        self._log_audit_event("SYSTEM_INIT", "Emergency shutdown handler initialized")

    # =========================================================================
    # AUDIT LOGGING
    # =========================================================================

    def _log_audit_event(
        self,
        event_type: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log an immutable audit event with provenance hash.

        Args:
            event_type: Type of audit event
            description: Human-readable description
            details: Additional event details

        Returns:
            Provenance hash of the audit entry
        """
        timestamp = datetime.now(timezone.utc)
        entry = {
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "description": description,
            "details": details or {},
        }

        # Calculate provenance hash
        content = f"{timestamp.isoformat()}|{event_type}|{description}"
        provenance_hash = hashlib.sha256(content.encode()).hexdigest()
        entry["provenance_hash"] = provenance_hash

        self._audit_log.append(entry)

        logger.debug(f"Audit event: {event_type} - {description}")

        return provenance_hash

    # =========================================================================
    # CONDITION DETECTION
    # =========================================================================

    def evaluate_conditions(
        self,
        measurements: List[ThermalMeasurement],
    ) -> List[ShutdownCondition]:
        """
        Evaluate thermal measurements for emergency conditions.

        Uses deterministic logic only - no ML inference for safety decisions.
        All detected conditions include full provenance tracking.

        Args:
            measurements: List of thermal measurements to evaluate

        Returns:
            List of detected shutdown conditions
        """
        with self._lock:
            conditions: List[ShutdownCondition] = []
            now = datetime.now(timezone.utc)

            for measurement in measurements:
                # Update measurement history
                self._last_measurements[measurement.asset_id] = measurement

                # Update temperature history for runaway detection
                if measurement.asset_id not in self._temperature_history:
                    self._temperature_history[measurement.asset_id] = deque(maxlen=100)
                self._temperature_history[measurement.asset_id].append(
                    (measurement.timestamp, measurement.surface_temp_c)
                )

                # Check burn risk (ASTM C1055)
                burn_condition = self._check_burn_risk(measurement)
                if burn_condition:
                    conditions.append(burn_condition)

                # Check insulation failure
                failure_condition = self._check_insulation_failure(measurement)
                if failure_condition:
                    conditions.append(failure_condition)

                # Check high heat loss
                heat_loss_condition = self._check_high_heat_loss(measurement)
                if heat_loss_condition:
                    conditions.append(heat_loss_condition)

                # Check thermal runaway
                runaway_condition = self._check_thermal_runaway(measurement)
                if runaway_condition:
                    conditions.append(runaway_condition)

                # Check OOD readings
                ood_condition = self._check_out_of_distribution(measurement)
                if ood_condition:
                    conditions.append(ood_condition)

            # Check for multiple hot spots
            hot_spot_condition = self._check_multiple_hot_spots(measurements)
            if hot_spot_condition:
                conditions.append(hot_spot_condition)

            # Check system integrity (active condition count)
            integrity_condition = self._check_system_integrity()
            if integrity_condition:
                conditions.append(integrity_condition)

            # Log audit event
            if conditions:
                self._log_audit_event(
                    "CONDITIONS_DETECTED",
                    f"Detected {len(conditions)} emergency conditions",
                    {"condition_types": [c.condition_type.value for c in conditions]}
                )

            return conditions

    def _check_burn_risk(
        self,
        measurement: ThermalMeasurement,
    ) -> Optional[ShutdownCondition]:
        """
        Check for burn risk per ASTM C1055.

        ASTM C1055 specifies 55C (131F) as the threshold for burn risk
        on surfaces that may be touched by personnel.
        """
        if measurement.surface_temp_c < self.config.burn_temp_threshold_c:
            return None

        # Determine severity based on temperature
        if measurement.surface_temp_c >= 80.0:
            severity = ShutdownLevel.SHUTDOWN
            actions = [
                ResponseAction.RESTRICT_ACCESS,
                ResponseAction.TEMPORARY_BARRIER,
                ResponseAction.EMERGENCY_REPAIR,
                ResponseAction.NOTIFY_OPERATOR,
            ]
        elif measurement.surface_temp_c >= 65.0:
            severity = ShutdownLevel.LIMIT
            actions = [
                ResponseAction.RESTRICT_ACCESS,
                ResponseAction.NOTIFY_OPERATOR,
                ResponseAction.ESCALATE_ENGINEERING,
            ]
        elif measurement.surface_temp_c >= 55.0:
            severity = ShutdownLevel.PROTECT
            actions = [
                ResponseAction.NOTIFY_OPERATOR,
                ResponseAction.INCREASE_MONITORING,
            ]
        else:
            severity = ShutdownLevel.ALERT
            actions = [ResponseAction.CONTINUE_MONITORING]

        condition_id = f"BURN_{measurement.asset_id}_{int(datetime.now(timezone.utc).timestamp())}"

        return ShutdownCondition(
            condition_id=condition_id,
            condition_type=ConditionType.BURN_RISK,
            severity=severity,
            description=(
                f"BURN HAZARD: Asset {measurement.asset_id} surface temperature "
                f"{measurement.surface_temp_c:.1f}C ({measurement.surface_temp_c * 9/5 + 32:.1f}F) "
                f"exceeds personnel safety limit {self.config.burn_temp_threshold_c}C "
                f"({self.config.burn_temp_fahrenheit:.1f}F). "
                f"Risk of second-degree burns with prolonged contact."
            ),
            asset_id=measurement.asset_id,
            measurement_values={
                "surface_temp_c": measurement.surface_temp_c,
                "surface_temp_f": measurement.surface_temp_c * 9/5 + 32,
                "ambient_temp_c": measurement.ambient_temp_c,
                "threshold_c": self.config.burn_temp_threshold_c,
            },
            threshold_violated=f"surface_temp > {self.config.burn_temp_threshold_c}C (ASTM C1055)",
            recommended_actions=actions,
            standard_reference="ASTM C1055, OSHA 29 CFR 1910.132",
        )

    def _check_insulation_failure(
        self,
        measurement: ThermalMeasurement,
    ) -> Optional[ShutdownCondition]:
        """
        Check for catastrophic insulation failure.

        Insulation failure is indicated by surface temperature approaching
        process temperature, indicating loss of thermal resistance.
        """
        if measurement.process_temp_c <= 0:
            return None

        # Calculate temperature ratio (surface - ambient) / (process - ambient)
        if measurement.process_temp_c <= measurement.ambient_temp_c:
            return None

        temp_delta = measurement.surface_temp_c - measurement.ambient_temp_c
        max_delta = measurement.process_temp_c - measurement.ambient_temp_c

        if max_delta <= 0:
            return None

        failure_ratio = temp_delta / max_delta

        # Thresholds for failure detection
        if failure_ratio < 0.5:
            return None

        if failure_ratio >= 0.9:
            severity = ShutdownLevel.SHUTDOWN
            description = "CATASTROPHIC insulation failure - near complete loss of thermal resistance"
        elif failure_ratio >= 0.75:
            severity = ShutdownLevel.LIMIT
            description = "SEVERE insulation failure - major loss of thermal resistance"
        elif failure_ratio >= 0.5:
            severity = ShutdownLevel.PROTECT
            description = "SIGNIFICANT insulation damage - moderate loss of thermal resistance"
        else:
            return None

        condition_id = f"FAIL_{measurement.asset_id}_{int(datetime.now(timezone.utc).timestamp())}"

        return ShutdownCondition(
            condition_id=condition_id,
            condition_type=ConditionType.INSULATION_FAILURE,
            severity=severity,
            description=(
                f"{description}. Asset {measurement.asset_id}: "
                f"Surface temp {measurement.surface_temp_c:.1f}C is {failure_ratio*100:.0f}% "
                f"of process-to-ambient delta. Immediate inspection required."
            ),
            asset_id=measurement.asset_id,
            measurement_values={
                "surface_temp_c": measurement.surface_temp_c,
                "process_temp_c": measurement.process_temp_c,
                "ambient_temp_c": measurement.ambient_temp_c,
                "failure_ratio": failure_ratio,
            },
            threshold_violated=f"failure_ratio >= 0.5 ({failure_ratio:.2f})",
            recommended_actions=[
                ResponseAction.EMERGENCY_REPAIR,
                ResponseAction.ESCALATE_ENGINEERING,
                ResponseAction.NOTIFY_OPERATOR,
            ],
            standard_reference="CINI Manual, ISO 12241",
        )

    def _check_high_heat_loss(
        self,
        measurement: ThermalMeasurement,
    ) -> Optional[ShutdownCondition]:
        """Check for excessive heat loss."""
        if measurement.heat_loss_w_m2 < self.config.high_heat_loss_threshold_w_m2:
            return None

        if measurement.heat_loss_w_m2 >= self.config.critical_heat_loss_threshold_w_m2:
            severity = ShutdownLevel.LIMIT
            actions = [
                ResponseAction.ESCALATE_ENGINEERING,
                ResponseAction.EMERGENCY_REPAIR,
                ResponseAction.NOTIFY_OPERATOR,
            ]
        else:
            severity = ShutdownLevel.PROTECT
            actions = [
                ResponseAction.NOTIFY_OPERATOR,
                ResponseAction.INCREASE_MONITORING,
            ]

        condition_id = f"HLOSS_{measurement.asset_id}_{int(datetime.now(timezone.utc).timestamp())}"

        return ShutdownCondition(
            condition_id=condition_id,
            condition_type=ConditionType.HIGH_HEAT_LOSS,
            severity=severity,
            description=(
                f"HIGH HEAT LOSS: Asset {measurement.asset_id} heat loss "
                f"{measurement.heat_loss_w_m2:.0f} W/m2 exceeds threshold "
                f"{self.config.high_heat_loss_threshold_w_m2:.0f} W/m2. "
                f"Significant energy waste and potential insulation damage."
            ),
            asset_id=measurement.asset_id,
            measurement_values={
                "heat_loss_w_m2": measurement.heat_loss_w_m2,
                "threshold_w_m2": self.config.high_heat_loss_threshold_w_m2,
            },
            threshold_violated=f"heat_loss > {self.config.high_heat_loss_threshold_w_m2} W/m2",
            recommended_actions=actions,
            standard_reference="ASTM C680",
        )

    def _check_thermal_runaway(
        self,
        measurement: ThermalMeasurement,
    ) -> Optional[ShutdownCondition]:
        """
        Check for thermal runaway conditions.

        Thermal runaway is indicated by rapid temperature rise over time,
        which may indicate equipment malfunction or fire risk.
        """
        history = self._temperature_history.get(measurement.asset_id)
        if not history or len(history) < 3:
            return None

        # Calculate rate of temperature change over the window
        window_minutes = self.config.thermal_runaway_window_minutes
        cutoff_time = measurement.timestamp - timedelta(minutes=window_minutes)

        recent_readings = [
            (ts, temp) for ts, temp in history
            if ts >= cutoff_time
        ]

        if len(recent_readings) < 2:
            return None

        # Calculate temperature change rate (C/min)
        first_ts, first_temp = recent_readings[0]
        last_ts, last_temp = recent_readings[-1]

        time_delta_minutes = (last_ts - first_ts).total_seconds() / 60
        if time_delta_minutes < 1:
            return None

        temp_change = last_temp - first_temp
        rate_c_per_min = temp_change / time_delta_minutes

        if rate_c_per_min < self.config.thermal_runaway_rate_c_per_min:
            return None

        # Thermal runaway detected
        if rate_c_per_min >= self.config.thermal_runaway_rate_c_per_min * 2:
            severity = ShutdownLevel.SHUTDOWN
        else:
            severity = ShutdownLevel.LIMIT

        condition_id = f"RUNAWAY_{measurement.asset_id}_{int(datetime.now(timezone.utc).timestamp())}"

        return ShutdownCondition(
            condition_id=condition_id,
            condition_type=ConditionType.THERMAL_RUNAWAY,
            severity=severity,
            description=(
                f"THERMAL RUNAWAY: Asset {measurement.asset_id} temperature rising at "
                f"{rate_c_per_min:.1f}C/min (threshold: {self.config.thermal_runaway_rate_c_per_min}C/min). "
                f"Temperature increased from {first_temp:.1f}C to {last_temp:.1f}C in "
                f"{time_delta_minutes:.1f} minutes. IMMEDIATE investigation required."
            ),
            asset_id=measurement.asset_id,
            measurement_values={
                "rate_c_per_min": rate_c_per_min,
                "start_temp_c": first_temp,
                "end_temp_c": last_temp,
                "time_window_min": time_delta_minutes,
                "threshold_rate": self.config.thermal_runaway_rate_c_per_min,
            },
            threshold_violated=f"rate > {self.config.thermal_runaway_rate_c_per_min} C/min",
            recommended_actions=[
                ResponseAction.ISOLATE_AREA,
                ResponseAction.PROCESS_REDUCTION,
                ResponseAction.ESCALATE_ENGINEERING,
                ResponseAction.NOTIFY_OPERATOR,
            ],
            standard_reference="NFPA 86, Plant Fire Safety Procedures",
        )

    def _check_out_of_distribution(
        self,
        measurement: ThermalMeasurement,
    ) -> Optional[ShutdownCondition]:
        """
        Check for out-of-distribution sensor readings.

        OOD readings may indicate sensor malfunction, calibration issues,
        or genuinely anomalous conditions requiring investigation.
        """
        # Physical impossibility checks
        issues = []

        # Surface temp below ambient (impossible for hot equipment)
        if measurement.surface_temp_c < measurement.ambient_temp_c - 5:
            issues.append(
                f"Surface temp {measurement.surface_temp_c:.1f}C below ambient "
                f"{measurement.ambient_temp_c:.1f}C"
            )

        # Surface temp above process temp (impossible)
        if measurement.process_temp_c > 0:
            if measurement.surface_temp_c > measurement.process_temp_c + 10:
                issues.append(
                    f"Surface temp {measurement.surface_temp_c:.1f}C exceeds "
                    f"process temp {measurement.process_temp_c:.1f}C"
                )

        # Extreme temperature values
        if measurement.surface_temp_c > 500 or measurement.surface_temp_c < -50:
            issues.append(
                f"Extreme temperature reading: {measurement.surface_temp_c:.1f}C"
            )

        # Negative heat loss (impossible)
        if measurement.heat_loss_w_m2 < 0:
            issues.append(
                f"Negative heat loss: {measurement.heat_loss_w_m2:.1f} W/m2"
            )

        if not issues:
            return None

        condition_id = f"OOD_{measurement.asset_id}_{int(datetime.now(timezone.utc).timestamp())}"

        return ShutdownCondition(
            condition_id=condition_id,
            condition_type=ConditionType.OUT_OF_DISTRIBUTION,
            severity=ShutdownLevel.PROTECT,
            description=(
                f"OUT-OF-DISTRIBUTION READING: Asset {measurement.asset_id} has "
                f"physically implausible sensor readings: {'; '.join(issues)}. "
                f"Verify sensor calibration and data integrity."
            ),
            asset_id=measurement.asset_id,
            measurement_values={
                "surface_temp_c": measurement.surface_temp_c,
                "ambient_temp_c": measurement.ambient_temp_c,
                "process_temp_c": measurement.process_temp_c,
                "heat_loss_w_m2": measurement.heat_loss_w_m2,
            },
            threshold_violated="Physical plausibility check failed",
            recommended_actions=[
                ResponseAction.INCREASE_MONITORING,
                ResponseAction.NOTIFY_OPERATOR,
            ],
            standard_reference="IR Thermography Standards, Sensor Calibration SOP",
        )

    def _check_multiple_hot_spots(
        self,
        measurements: List[ThermalMeasurement],
    ) -> Optional[ShutdownCondition]:
        """Check for multiple concurrent hot spots."""
        # Count measurements exceeding burn threshold
        hot_spots = [
            m for m in measurements
            if m.surface_temp_c >= self.config.burn_temp_threshold_c
        ]

        if len(hot_spots) < self.config.hot_spot_count_threshold:
            return None

        if len(hot_spots) >= self.config.hot_spot_count_threshold * 2:
            severity = ShutdownLevel.SHUTDOWN
        elif len(hot_spots) >= self.config.hot_spot_count_threshold * 1.5:
            severity = ShutdownLevel.LIMIT
        else:
            severity = ShutdownLevel.PROTECT

        asset_ids = [m.asset_id for m in hot_spots]
        avg_temp = sum(m.surface_temp_c for m in hot_spots) / len(hot_spots)
        max_temp = max(m.surface_temp_c for m in hot_spots)

        condition_id = f"HOTSPOTS_{int(datetime.now(timezone.utc).timestamp())}"

        return ShutdownCondition(
            condition_id=condition_id,
            condition_type=ConditionType.MULTIPLE_HOT_SPOTS,
            severity=severity,
            description=(
                f"MULTIPLE HOT SPOTS: {len(hot_spots)} locations exceed burn threshold "
                f"(limit: {self.config.hot_spot_count_threshold}). "
                f"Average temp: {avg_temp:.1f}C, Max temp: {max_temp:.1f}C. "
                f"May indicate systemic insulation failure or process upset."
            ),
            measurement_values={
                "hot_spot_count": len(hot_spots),
                "threshold": self.config.hot_spot_count_threshold,
                "average_temp_c": avg_temp,
                "max_temp_c": max_temp,
            },
            threshold_violated=f"hot_spot_count >= {self.config.hot_spot_count_threshold}",
            recommended_actions=[
                ResponseAction.RESTRICT_ACCESS,
                ResponseAction.ESCALATE_ENGINEERING,
                ResponseAction.NOTIFY_OPERATOR,
                ResponseAction.PROCESS_REDUCTION,
            ],
            standard_reference="Plant Insulation Integrity SOP",
        )

    def _check_system_integrity(self) -> Optional[ShutdownCondition]:
        """Check overall system integrity based on active condition count."""
        active_count = len(self._active_events)

        if active_count < self.config.max_active_conditions:
            return None

        condition_id = f"SYSINT_{int(datetime.now(timezone.utc).timestamp())}"

        return ShutdownCondition(
            condition_id=condition_id,
            condition_type=ConditionType.SYSTEM_INTEGRITY,
            severity=ShutdownLevel.LIMIT,
            description=(
                f"SYSTEM INTEGRITY ALERT: {active_count} active conditions exceed "
                f"system threshold of {self.config.max_active_conditions}. "
                f"System may be overwhelmed or experiencing widespread issues."
            ),
            measurement_values={
                "active_count": active_count,
                "threshold": self.config.max_active_conditions,
            },
            threshold_violated=f"active_conditions >= {self.config.max_active_conditions}",
            recommended_actions=[
                ResponseAction.ESCALATE_ENGINEERING,
                ResponseAction.SUSPEND_RECOMMENDATIONS,
            ],
            standard_reference="System Capacity Limits",
        )

    # =========================================================================
    # SHUTDOWN TRIGGERING
    # =========================================================================

    def trigger_shutdown(
        self,
        conditions: List[ShutdownCondition],
    ) -> ShutdownResult:
        """
        Trigger appropriate shutdown level based on detected conditions.

        This method creates events, sends notifications, and updates system
        state based on the severity of detected conditions. Note that this
        is ADVISORY ONLY - no direct control actions are taken.

        Args:
            conditions: List of detected shutdown conditions

        Returns:
            ShutdownResult with details of actions taken
        """
        with self._lock:
            if not conditions:
                return ShutdownResult(
                    success=True,
                    shutdown_level=ShutdownLevel.ALERT,
                    message="No conditions to process",
                )

            # Determine highest severity
            max_severity = max(c.severity for c in conditions)
            events_created = []
            notifications_sent = []

            for condition in conditions:
                # Create event for this condition
                event = self._create_event(condition)
                events_created.append(event.event_id)

                # Send notification if callback registered
                if self._notification_callback:
                    try:
                        self._notification_callback(event, condition.severity)
                        notifications_sent.append(f"callback:{event.event_id}")
                    except Exception as e:
                        logger.error(f"Notification callback failed: {e}")

            # Update system state based on max severity
            self._current_shutdown_level = max_severity

            # Suspend recommendations if Level 4
            if max_severity >= ShutdownLevel.SHUTDOWN:
                self._recommendations_suspended = True
                self._manual_override_active = True

                self._log_audit_event(
                    "RECOMMENDATIONS_SUSPENDED",
                    f"Recommendations suspended due to Level {max_severity.value} condition",
                    {"conditions": [c.condition_id for c in conditions]}
                )

            result = ShutdownResult(
                success=True,
                shutdown_level=max_severity,
                events_created=events_created,
                notifications_sent=notifications_sent,
                recommendations_suspended=self._recommendations_suspended,
                manual_override_required=(max_severity >= ShutdownLevel.SHUTDOWN),
                message=f"Processed {len(conditions)} conditions, max severity: Level {max_severity.value}",
            )

            self._log_audit_event(
                "SHUTDOWN_TRIGGERED",
                f"Shutdown Level {max_severity.value} triggered",
                {
                    "conditions_count": len(conditions),
                    "events_created": events_created,
                    "recommendations_suspended": self._recommendations_suspended,
                }
            )

            return result

    def _create_event(self, condition: ShutdownCondition) -> ShutdownEvent:
        """Create a shutdown event from a condition."""
        event_id = f"EVT_{uuid.uuid4().hex[:12]}"

        event = ShutdownEvent(
            event_id=event_id,
            condition=condition,
            state=ConditionState.ACTIVE,
            escalation_level=EscalationLevel.OPERATOR,
        )

        self._active_events[event_id] = event

        # Track condition history
        if condition.asset_id:
            if condition.asset_id not in self._condition_history:
                self._condition_history[condition.asset_id] = []
            self._condition_history[condition.asset_id].append(condition)

        logger.warning(
            f"Emergency event created: id={event_id}, "
            f"type={condition.condition_type.value}, "
            f"severity=Level {condition.severity.value}"
        )

        return event

    # =========================================================================
    # CONDITION ACKNOWLEDGMENT AND RESOLUTION
    # =========================================================================

    def acknowledge_condition(
        self,
        condition_id: str,
        operator_id: str,
        notes: str = "",
    ) -> bool:
        """
        Acknowledge an active condition.

        Acknowledgment indicates an operator has reviewed the condition
        and is taking appropriate action. Does not clear the condition.

        Args:
            condition_id: Condition identifier (or event_id)
            operator_id: ID of acknowledging operator
            notes: Optional acknowledgment notes

        Returns:
            True if acknowledged successfully, False if not found
        """
        with self._lock:
            # Find event by condition_id or event_id
            event = None
            for e in self._active_events.values():
                if e.event_id == condition_id or e.condition.condition_id == condition_id:
                    event = e
                    break

            if event is None:
                logger.warning(f"Condition/event {condition_id} not found for acknowledgment")
                return False

            if event.state not in (ConditionState.ACTIVE, ConditionState.ESCALATED):
                logger.warning(f"Event {event.event_id} is not active (state={event.state.value})")
                return False

            event.state = ConditionState.ACKNOWLEDGED
            event.acknowledged_at = datetime.now(timezone.utc)
            event.acknowledged_by = operator_id

            self._log_audit_event(
                "CONDITION_ACKNOWLEDGED",
                f"Condition acknowledged by {operator_id}",
                {
                    "event_id": event.event_id,
                    "condition_id": event.condition.condition_id,
                    "notes": notes,
                }
            )

            logger.info(f"Event {event.event_id} acknowledged by {operator_id}")

            return True

    def resolve_condition(
        self,
        condition_id: str,
        operator_id: str,
        resolution_notes: str = "",
    ) -> bool:
        """
        Resolve an active condition.

        Resolution clears the condition and moves it to resolved state.
        Requires manual verification that the underlying issue is fixed.

        Args:
            condition_id: Condition identifier (or event_id)
            operator_id: ID of resolving operator
            resolution_notes: Description of resolution

        Returns:
            True if resolved successfully, False if not found
        """
        with self._lock:
            # Find event
            event = None
            event_key = None
            for key, e in self._active_events.items():
                if e.event_id == condition_id or e.condition.condition_id == condition_id:
                    event = e
                    event_key = key
                    break

            if event is None:
                logger.warning(f"Condition/event {condition_id} not found for resolution")
                return False

            event.state = ConditionState.RESOLVED
            event.resolved_at = datetime.now(timezone.utc)
            event.resolved_by = operator_id
            event.resolution_notes = resolution_notes

            # Move to resolved list
            self._resolved_events.append(event)
            del self._active_events[event_key]

            # Check if we can resume recommendations
            self._check_resume_recommendations()

            self._log_audit_event(
                "CONDITION_RESOLVED",
                f"Condition resolved by {operator_id}",
                {
                    "event_id": event.event_id,
                    "condition_id": event.condition.condition_id,
                    "resolution_notes": resolution_notes,
                }
            )

            logger.info(f"Event {event.event_id} resolved by {operator_id}")

            return True

    def _check_resume_recommendations(self) -> None:
        """Check if recommendations can be resumed."""
        if not self._recommendations_suspended:
            return

        # Check if any Level 4 conditions remain
        level4_active = any(
            e.condition.severity >= ShutdownLevel.SHUTDOWN
            for e in self._active_events.values()
        )

        if not level4_active:
            self._recommendations_suspended = False
            self._manual_override_active = False

            # Update current shutdown level
            if self._active_events:
                self._current_shutdown_level = max(
                    e.condition.severity for e in self._active_events.values()
                )
            else:
                self._current_shutdown_level = ShutdownLevel.ALERT

            self._log_audit_event(
                "RECOMMENDATIONS_RESUMED",
                "Recommendations resumed after Level 4 conditions resolved"
            )

            logger.info("Recommendations resumed - no Level 4 conditions active")

    def clear_manual_override(
        self,
        operator_id: str,
        authorization_code: str,
    ) -> bool:
        """
        Clear manual override to resume normal operations.

        Requires explicit authorization to clear override state.

        Args:
            operator_id: ID of operator clearing override
            authorization_code: Authorization code for override clear

        Returns:
            True if override cleared, False if denied
        """
        with self._lock:
            if not self._manual_override_active:
                logger.info("No manual override active")
                return True

            # In production, verify authorization_code against auth system
            # For now, require a non-empty code
            if not authorization_code:
                logger.warning(f"Override clear denied - no authorization code")
                return False

            self._manual_override_active = False
            self._check_resume_recommendations()

            self._log_audit_event(
                "MANUAL_OVERRIDE_CLEARED",
                f"Manual override cleared by {operator_id}",
                {"authorization_code_provided": bool(authorization_code)}
            )

            logger.info(f"Manual override cleared by {operator_id}")

            return True

    # =========================================================================
    # ESCALATION
    # =========================================================================

    def check_auto_escalation(self) -> List[ShutdownEvent]:
        """
        Check for events that need automatic escalation.

        Events are escalated if not acknowledged within the configured
        timeout period for their severity level.

        Returns:
            List of escalated events
        """
        with self._lock:
            escalated = []
            now = datetime.now(timezone.utc)

            for event in self._active_events.values():
                if event.state != ConditionState.ACTIVE:
                    continue

                # Determine escalation timeout based on severity
                if event.condition.severity == ShutdownLevel.SHUTDOWN:
                    timeout = self.config.level4_escalation_minutes
                elif event.condition.severity == ShutdownLevel.LIMIT:
                    timeout = self.config.level3_escalation_minutes
                elif event.condition.severity == ShutdownLevel.PROTECT:
                    timeout = self.config.level2_escalation_minutes
                else:
                    timeout = self.config.escalation_timeout_minutes

                if event.age_minutes >= timeout:
                    self._escalate_event(event)
                    escalated.append(event)

            return escalated

    def _escalate_event(self, event: ShutdownEvent) -> None:
        """Escalate an event to the next level."""
        escalation_order = [
            EscalationLevel.NONE,
            EscalationLevel.OPERATOR,
            EscalationLevel.SUPERVISOR,
            EscalationLevel.ENGINEER,
            EscalationLevel.SAFETY_TEAM,
            EscalationLevel.PLANT_MANAGER,
        ]

        try:
            current_idx = escalation_order.index(event.escalation_level)
            if current_idx < len(escalation_order) - 1:
                new_level = escalation_order[current_idx + 1]
            else:
                new_level = event.escalation_level
        except ValueError:
            new_level = EscalationLevel.SUPERVISOR

        old_level = event.escalation_level
        event.escalation_level = new_level
        event.state = ConditionState.ESCALATED

        event.escalation_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from_level": old_level.value,
            "to_level": new_level.value,
            "reason": "Automatic escalation - unacknowledged timeout",
        })

        # Invoke escalation callback
        if self._escalation_callback:
            try:
                self._escalation_callback(event, new_level)
            except Exception as e:
                logger.error(f"Escalation callback failed: {e}")

        self._log_audit_event(
            "EVENT_ESCALATED",
            f"Event {event.event_id} escalated from {old_level.value} to {new_level.value}",
            {
                "event_id": event.event_id,
                "condition_type": event.condition.condition_type.value,
                "severity": event.condition.severity.value,
            }
        )

        logger.warning(
            f"Event {event.event_id} escalated: {old_level.value} -> {new_level.value}"
        )

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_active_conditions(self) -> List[ShutdownCondition]:
        """
        Get all currently active shutdown conditions.

        Returns:
            List of active shutdown conditions
        """
        with self._lock:
            return [e.condition for e in self._active_events.values()]

    def get_active_events(
        self,
        severity: Optional[ShutdownLevel] = None,
        condition_type: Optional[ConditionType] = None,
    ) -> List[ShutdownEvent]:
        """
        Get active events with optional filtering.

        Args:
            severity: Filter by severity level
            condition_type: Filter by condition type

        Returns:
            List of matching active events
        """
        with self._lock:
            events = list(self._active_events.values())

            if severity:
                events = [e for e in events if e.condition.severity == severity]

            if condition_type:
                events = [e for e in events if e.condition.condition_type == condition_type]

            return sorted(events, key=lambda e: e.created_at, reverse=True)

    def get_event_by_id(self, event_id: str) -> Optional[ShutdownEvent]:
        """Get a specific event by ID."""
        with self._lock:
            return self._active_events.get(event_id)

    def get_condition_history(
        self,
        asset_id: str,
        limit: int = 100,
    ) -> List[ShutdownCondition]:
        """
        Get condition history for a specific asset.

        Args:
            asset_id: Asset identifier
            limit: Maximum conditions to return

        Returns:
            List of historical conditions for the asset
        """
        with self._lock:
            history = self._condition_history.get(asset_id, [])
            return list(reversed(history[-limit:]))

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall emergency shutdown system status.

        Returns:
            Status dictionary with system state
        """
        with self._lock:
            # Count by severity
            severity_counts = {level.name: 0 for level in ShutdownLevel}
            for event in self._active_events.values():
                severity_counts[event.condition.severity.name] += 1

            # Count by type
            type_counts: Dict[str, int] = {}
            for event in self._active_events.values():
                t = event.condition.condition_type.value
                type_counts[t] = type_counts.get(t, 0) + 1

            # Determine overall status
            if self._recommendations_suspended:
                status = "SHUTDOWN"
            elif severity_counts.get("LIMIT", 0) > 0:
                status = "LIMITED"
            elif severity_counts.get("PROTECT", 0) > 0:
                status = "PROTECTED"
            elif sum(severity_counts.values()) > 0:
                status = "ALERT"
            else:
                status = "NORMAL"

            return {
                "status": status,
                "current_shutdown_level": self._current_shutdown_level.value,
                "current_shutdown_level_name": self._current_shutdown_level.name,
                "recommendations_suspended": self._recommendations_suspended,
                "manual_override_active": self._manual_override_active,
                "total_active_events": len(self._active_events),
                "events_by_severity": severity_counts,
                "events_by_type": type_counts,
                "unacknowledged_count": sum(
                    1 for e in self._active_events.values()
                    if e.state == ConditionState.ACTIVE
                ),
                "escalated_count": sum(
                    1 for e in self._active_events.values()
                    if e.state == ConditionState.ESCALATED
                ),
                "resolved_last_24h": sum(
                    1 for e in self._resolved_events
                    if e.resolved_at and
                    e.resolved_at > datetime.now(timezone.utc) - timedelta(hours=24)
                ),
                "audit_log_entries": len(self._audit_log),
                "config": {
                    "burn_temp_threshold_c": self.config.burn_temp_threshold_c,
                    "hot_spot_count_threshold": self.config.hot_spot_count_threshold,
                    "thermal_runaway_rate_c_per_min": self.config.thermal_runaway_rate_c_per_min,
                    "auto_shutdown_enabled": self.config.auto_shutdown_enabled,
                },
            }

    def get_audit_log(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get audit log entries.

        Args:
            limit: Maximum entries to return
            event_type: Filter by event type

        Returns:
            List of audit log entries
        """
        with self._lock:
            entries = list(self._audit_log)

            if event_type:
                entries = [e for e in entries if e.get("event_type") == event_type]

            return list(reversed(entries[-limit:]))

    # =========================================================================
    # RESET AND MAINTENANCE
    # =========================================================================

    def reset(
        self,
        operator_id: str,
        authorization_code: str,
    ) -> bool:
        """
        Reset all active conditions and system state.

        WARNING: This clears all active conditions. Use only when appropriate.

        Args:
            operator_id: ID of operator performing reset
            authorization_code: Authorization code for reset

        Returns:
            True if reset successful
        """
        with self._lock:
            if not authorization_code:
                logger.warning(f"Reset denied - no authorization code")
                return False

            # Move all active events to resolved
            for event in self._active_events.values():
                event.state = ConditionState.RESOLVED
                event.resolved_at = datetime.now(timezone.utc)
                event.resolved_by = operator_id
                event.resolution_notes = "System reset"
                self._resolved_events.append(event)

            self._active_events.clear()
            self._recommendations_suspended = False
            self._manual_override_active = False
            self._current_shutdown_level = ShutdownLevel.ALERT

            self._log_audit_event(
                "SYSTEM_RESET",
                f"System reset by {operator_id}",
                {"authorization_code_provided": True}
            )

            logger.warning(f"Emergency shutdown system reset by {operator_id}")

            return True


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def check_burn_risk_quick(
    surface_temp_c: float,
    threshold_c: float = 55.0,
) -> Tuple[bool, ShutdownLevel]:
    """
    Quick check for burn risk without full handler.

    Args:
        surface_temp_c: Surface temperature in Celsius
        threshold_c: Burn threshold (default: 55C per ASTM C1055)

    Returns:
        Tuple of (is_burn_risk, severity_level)
    """
    if surface_temp_c < threshold_c:
        return False, ShutdownLevel.ALERT

    if surface_temp_c >= 80.0:
        return True, ShutdownLevel.SHUTDOWN
    elif surface_temp_c >= 65.0:
        return True, ShutdownLevel.LIMIT
    elif surface_temp_c >= 55.0:
        return True, ShutdownLevel.PROTECT
    else:
        return True, ShutdownLevel.ALERT


def calculate_thermal_runaway_rate(
    temperatures: List[Tuple[datetime, float]],
) -> float:
    """
    Calculate temperature change rate from historical data.

    Args:
        temperatures: List of (timestamp, temperature) tuples

    Returns:
        Temperature change rate in C/min
    """
    if len(temperatures) < 2:
        return 0.0

    first_ts, first_temp = temperatures[0]
    last_ts, last_temp = temperatures[-1]

    time_delta_minutes = (last_ts - first_ts).total_seconds() / 60
    if time_delta_minutes < 0.1:
        return 0.0

    return (last_temp - first_temp) / time_delta_minutes


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ShutdownLevel",
    "ConditionType",
    "ConditionState",
    "EscalationLevel",
    "ResponseAction",
    # Data models
    "ThermalMeasurement",
    "ShutdownCondition",
    "ShutdownEvent",
    "ShutdownResult",
    # Configuration
    "EmergencyShutdownConfig",
    # Main class
    "InsulationEmergencyShutdown",
    # Convenience functions
    "check_burn_risk_quick",
    "calculate_thermal_runaway_rate",
]
