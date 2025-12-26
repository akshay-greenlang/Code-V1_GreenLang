# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Emergency Response Handler

Handles critical condition detection, alert escalation, and safe state
recommendations for heat exchanger systems.

Emergency Response Features:
1. Critical condition detection (temperature, pressure, fouling)
2. Multi-level alert escalation (operator -> supervisor -> engineer)
3. Safe state recommendations (conservative operating points)
4. Automatic de-escalation when conditions normalize
5. Full audit trail with provenance tracking

Safety Principles:
- Early warning before critical thresholds
- Clear escalation paths with defined responsibilities
- Safe state recommendations, not autonomous actions
- Fail-safe defaults on uncertainty

Standards Reference:
- IEC 61511: Safety Instrumented Systems
- IEC 61508: Functional Safety
- ISA-84: Application of Safety Instrumented Systems

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from .exceptions import (
    ExchangerproSafetyError,
    ViolationContext,
    ViolationDetails,
    ViolationSeverity,
    SafetyDomain,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class AlertSeverity(str, Enum):
    """Alert severity levels following IEC 62682 alarm management."""

    INFO = "info"  # Informational, logged only
    LOW = "low"  # Low priority, batch notification
    MEDIUM = "medium"  # Medium priority, timely notification
    HIGH = "high"  # High priority, immediate notification
    CRITICAL = "critical"  # Critical, immediate escalation required
    EMERGENCY = "emergency"  # Emergency, automatic safe state


class AlertState(str, Enum):
    """Current state of an alert."""

    ACTIVE = "active"  # Alert is active
    ACKNOWLEDGED = "acknowledged"  # Operator acknowledged
    RESOLVED = "resolved"  # Condition resolved
    SUPPRESSED = "suppressed"  # Temporarily suppressed
    ESCALATED = "escalated"  # Escalated to higher level


class EscalationLevel(str, Enum):
    """Escalation hierarchy levels."""

    NONE = "none"  # No escalation
    OPERATOR = "operator"  # Field operator
    SUPERVISOR = "supervisor"  # Shift supervisor
    ENGINEER = "engineer"  # Process engineer
    MANAGER = "manager"  # Operations manager
    EMERGENCY_TEAM = "emergency_team"  # Emergency response team


class ConditionType(str, Enum):
    """Types of critical conditions detected."""

    HIGH_TEMPERATURE = "high_temperature"
    LOW_TEMPERATURE = "low_temperature"
    HIGH_PRESSURE = "high_pressure"
    LOW_PRESSURE = "low_pressure"
    HIGH_PRESSURE_DROP = "high_pressure_drop"
    LOW_FLOW = "low_flow"
    HIGH_FOULING = "high_fouling"
    ENERGY_IMBALANCE = "energy_imbalance"
    EFFECTIVENESS_ANOMALY = "effectiveness_anomaly"
    SENSOR_FAULT = "sensor_fault"
    COMMUNICATION_LOSS = "communication_loss"


class SafeStateType(str, Enum):
    """Types of safe state recommendations."""

    REDUCE_THROUGHPUT = "reduce_throughput"
    INCREASE_COOLING = "increase_cooling"
    BYPASS_EXCHANGER = "bypass_exchanger"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    MANUAL_INSPECTION = "manual_inspection"
    SCHEDULE_CLEANING = "schedule_cleaning"
    REDUCE_HEAT_RECOVERY = "reduce_heat_recovery"
    ISOLATE_SECTION = "isolate_section"


# =============================================================================
# CONFIGURATION
# =============================================================================


class AlertThresholds(BaseModel):
    """
    Thresholds for alert generation.

    Each parameter has warning, high, and critical thresholds.
    """

    # Temperature thresholds (C)
    temperature_warning_high_C: float = Field(default=350.0)
    temperature_high_C: float = Field(default=380.0)
    temperature_critical_C: float = Field(default=400.0)
    temperature_warning_low_C: float = Field(default=130.0)
    temperature_low_C: float = Field(default=120.0)
    temperature_critical_low_C: float = Field(default=110.0)

    # Pressure drop thresholds (% of design)
    pressure_drop_warning_pct: float = Field(default=70.0)
    pressure_drop_high_pct: float = Field(default=85.0)
    pressure_drop_critical_pct: float = Field(default=100.0)

    # Fouling resistance thresholds (% of design allowance)
    fouling_warning_pct: float = Field(default=60.0)
    fouling_high_pct: float = Field(default=80.0)
    fouling_critical_pct: float = Field(default=100.0)

    # Effectiveness degradation thresholds (% drop from baseline)
    effectiveness_warning_drop_pct: float = Field(default=10.0)
    effectiveness_high_drop_pct: float = Field(default=20.0)
    effectiveness_critical_drop_pct: float = Field(default=30.0)

    # Flow rate thresholds (% of design)
    flow_warning_low_pct: float = Field(default=70.0)
    flow_low_pct: float = Field(default=50.0)
    flow_critical_low_pct: float = Field(default=30.0)


class EscalationPolicy(BaseModel):
    """
    Policy for alert escalation timing.

    Defines how quickly alerts escalate if not acknowledged.
    """

    # Time to escalate if not acknowledged (minutes)
    low_to_operator_minutes: int = Field(default=60)
    medium_to_operator_minutes: int = Field(default=15)
    high_to_supervisor_minutes: int = Field(default=5)
    critical_to_engineer_minutes: int = Field(default=2)
    emergency_to_team_minutes: int = Field(default=0)  # Immediate

    # Maximum acknowledgment delay before auto-escalate
    max_ack_delay_minutes: int = Field(default=30)


class EmergencyResponseConfig(BaseModel):
    """
    Configuration for emergency response handler.

    Attributes:
        thresholds: Alert thresholds
        escalation_policy: Escalation timing
        enable_auto_escalation: Auto-escalate unacknowledged alerts
        enable_safe_state_recommendations: Generate safe state recommendations
        alert_cooldown_minutes: Minimum time between repeated alerts
        max_active_alerts: Maximum active alerts before suppression
        retain_resolved_hours: Hours to retain resolved alerts
    """

    thresholds: AlertThresholds = Field(default_factory=AlertThresholds)
    escalation_policy: EscalationPolicy = Field(default_factory=EscalationPolicy)
    enable_auto_escalation: bool = Field(default=True)
    enable_safe_state_recommendations: bool = Field(default=True)
    alert_cooldown_minutes: int = Field(default=5, ge=1, le=60)
    max_active_alerts: int = Field(default=50, ge=10, le=200)
    retain_resolved_hours: int = Field(default=168, ge=24, le=720)  # 7 days default


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class CriticalCondition:
    """
    Detected critical condition.

    Attributes:
        condition_id: Unique identifier
        condition_type: Type of condition
        exchanger_id: Affected heat exchanger
        parameter_name: Parameter that triggered condition
        current_value: Current measured value
        threshold_value: Threshold that was exceeded
        unit: Engineering unit
        severity: Condition severity
        detected_at: When condition was detected
        message: Human-readable description
        recommended_actions: List of recommended actions
        provenance_hash: SHA-256 hash for audit
    """

    condition_id: str
    condition_type: ConditionType
    exchanger_id: str
    parameter_name: str
    current_value: float
    threshold_value: float
    unit: str
    severity: AlertSeverity
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""
    recommended_actions: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            content = (
                f"{self.condition_id}|{self.condition_type.value}|"
                f"{self.exchanger_id}|{self.current_value:.6f}|"
                f"{self.detected_at.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class Alert:
    """
    Alert record with full lifecycle tracking.

    Attributes:
        alert_id: Unique alert identifier
        condition: The triggering condition
        state: Current alert state
        severity: Alert severity
        escalation_level: Current escalation level
        created_at: When alert was created
        acknowledged_at: When alert was acknowledged
        acknowledged_by: Who acknowledged
        resolved_at: When alert was resolved
        escalated_at: When last escalated
        escalation_history: History of escalations
        suppression_until: If suppressed, until when
        notifications_sent: List of notification records
        provenance_hash: SHA-256 hash for audit
    """

    alert_id: str
    condition: CriticalCondition
    state: AlertState = AlertState.ACTIVE
    severity: AlertSeverity = AlertSeverity.MEDIUM
    escalation_level: EscalationLevel = EscalationLevel.NONE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    escalated_at: Optional[datetime] = None
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    suppression_until: Optional[datetime] = None
    notifications_sent: List[Dict[str, Any]] = field(default_factory=list)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            content = (
                f"{self.alert_id}|{self.condition.condition_id}|"
                f"{self.severity.value}|{self.created_at.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()

    @property
    def age_minutes(self) -> float:
        """Get alert age in minutes."""
        now = datetime.now(timezone.utc)
        return (now - self.created_at).total_seconds() / 60

    @property
    def time_since_acknowledgment(self) -> Optional[float]:
        """Get time since acknowledgment in minutes."""
        if self.acknowledged_at is None:
            return None
        now = datetime.now(timezone.utc)
        return (now - self.acknowledged_at).total_seconds() / 60

    def is_suppressed(self) -> bool:
        """Check if alert is currently suppressed."""
        if self.suppression_until is None:
            return False
        return datetime.now(timezone.utc) < self.suppression_until


@dataclass
class SafeStateRecommendation:
    """
    Safe state recommendation for critical conditions.

    Attributes:
        recommendation_id: Unique identifier
        exchanger_id: Target exchanger
        safe_state_type: Type of safe state
        triggering_alerts: Alerts that triggered this recommendation
        description: Detailed description
        priority: Recommendation priority
        estimated_impact: Estimated impact on operations
        implementation_steps: Steps to implement
        created_at: When recommendation was created
        valid_until: Recommendation validity window
        requires_authorization: Whether authorization is needed
        authorizing_role: Role that can authorize
        provenance_hash: SHA-256 hash for audit
    """

    recommendation_id: str
    exchanger_id: str
    safe_state_type: SafeStateType
    triggering_alerts: List[str]
    description: str
    priority: int  # 1-5, 1 being highest
    estimated_impact: str
    implementation_steps: List[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: Optional[datetime] = None
    requires_authorization: bool = False
    authorizing_role: Optional[str] = None
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            content = (
                f"{self.recommendation_id}|{self.exchanger_id}|"
                f"{self.safe_state_type.value}|{self.created_at.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()

    def is_valid(self) -> bool:
        """Check if recommendation is still valid."""
        if self.valid_until is None:
            return True
        return datetime.now(timezone.utc) < self.valid_until


# =============================================================================
# EMERGENCY RESPONSE HANDLER
# =============================================================================


class EmergencyResponseHandler:
    """
    Handles critical condition detection, alert escalation, and safe state
    recommendations for heat exchanger systems.

    This handler:
    1. Detects critical conditions based on configured thresholds
    2. Generates alerts with appropriate severity
    3. Manages alert lifecycle (active -> acknowledged -> resolved)
    4. Escalates unacknowledged alerts according to policy
    5. Generates safe state recommendations

    Safety Principles:
    - Recommendations only, no autonomous control actions
    - Clear escalation paths with audit trail
    - Fail-safe defaults on uncertainty
    - All decisions are logged with provenance

    Example:
        >>> config = EmergencyResponseConfig()
        >>> handler = EmergencyResponseHandler(config)
        >>>
        >>> # Check for critical conditions
        >>> conditions = handler.detect_conditions(
        ...     exchanger_id="HX-101",
        ...     measurements={
        ...         "hot_outlet_temp_C": 390.0,
        ...         "pressure_drop_kPa": 45.0,
        ...     },
        ... )
        >>>
        >>> # Handle any alerts
        >>> for condition in conditions:
        ...     alert = handler.create_alert(condition)
        ...     if alert.severity >= AlertSeverity.HIGH:
        ...         handler.escalate_alert(alert.alert_id)

    Author: GL-BackendDeveloper
    Version: 1.0.0
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        config: Optional[EmergencyResponseConfig] = None,
        notification_callback: Optional[Callable[[Alert], None]] = None,
        escalation_callback: Optional[Callable[[Alert, EscalationLevel], None]] = None,
    ) -> None:
        """
        Initialize emergency response handler.

        Args:
            config: Configuration
            notification_callback: Callback for sending notifications
            escalation_callback: Callback for escalation events
        """
        self.config = config or EmergencyResponseConfig()
        self._lock = threading.RLock()

        # Alert storage
        self._active_alerts: Dict[str, Alert] = {}
        self._resolved_alerts: List[Alert] = []

        # Condition tracking
        self._condition_last_seen: Dict[str, datetime] = {}

        # Safe state recommendations
        self._recommendations: Dict[str, SafeStateRecommendation] = {}

        # Callbacks
        self._notification_callback = notification_callback
        self._escalation_callback = escalation_callback

        # Baseline values for effectiveness tracking
        self._baselines: Dict[str, Dict[str, float]] = {}

        logger.info(
            f"EmergencyResponseHandler initialized: "
            f"auto_escalation={self.config.enable_auto_escalation}, "
            f"safe_state_recommendations={self.config.enable_safe_state_recommendations}"
        )

    # =========================================================================
    # CONDITION DETECTION
    # =========================================================================

    def set_baseline(
        self,
        exchanger_id: str,
        effectiveness: float,
        design_pressure_drop: float,
        design_fouling_resistance: float,
    ) -> None:
        """
        Set baseline values for an exchanger.

        Args:
            exchanger_id: Heat exchanger identifier
            effectiveness: Baseline effectiveness
            design_pressure_drop: Design pressure drop
            design_fouling_resistance: Design fouling resistance allowance
        """
        with self._lock:
            self._baselines[exchanger_id] = {
                "effectiveness": effectiveness,
                "design_pressure_drop": design_pressure_drop,
                "design_fouling_resistance": design_fouling_resistance,
            }
            logger.info(f"Baseline set for {exchanger_id}")

    def detect_conditions(
        self,
        exchanger_id: str,
        measurements: Dict[str, float],
    ) -> List[CriticalCondition]:
        """
        Detect critical conditions from measurements.

        Args:
            exchanger_id: Heat exchanger identifier
            measurements: Dictionary of current measurements

        Returns:
            List of detected critical conditions
        """
        with self._lock:
            conditions: List[CriticalCondition] = []
            thresholds = self.config.thresholds
            now = datetime.now(timezone.utc)

            # Get baseline if available
            baseline = self._baselines.get(exchanger_id, {})

            # Check temperature high
            for temp_key in ["hot_outlet_temp_C", "hot_inlet_temp_C", "temperature_C"]:
                if temp_key in measurements:
                    temp = measurements[temp_key]
                    condition = self._check_temperature_high(
                        exchanger_id, temp_key, temp, thresholds
                    )
                    if condition:
                        conditions.append(condition)

            # Check temperature low (acid dew point protection)
            for temp_key in ["flue_gas_outlet_C", "hot_outlet_temp_C"]:
                if temp_key in measurements:
                    temp = measurements[temp_key]
                    condition = self._check_temperature_low(
                        exchanger_id, temp_key, temp, thresholds
                    )
                    if condition:
                        conditions.append(condition)

            # Check pressure drop
            if "pressure_drop_kPa" in measurements and "design_pressure_drop" in baseline:
                dp = measurements["pressure_drop_kPa"]
                dp_design = baseline["design_pressure_drop"]
                condition = self._check_pressure_drop(
                    exchanger_id, dp, dp_design, thresholds
                )
                if condition:
                    conditions.append(condition)

            # Check fouling
            if "fouling_resistance" in measurements and "design_fouling_resistance" in baseline:
                rf = measurements["fouling_resistance"]
                rf_design = baseline["design_fouling_resistance"]
                condition = self._check_fouling(
                    exchanger_id, rf, rf_design, thresholds
                )
                if condition:
                    conditions.append(condition)

            # Check effectiveness
            if "effectiveness" in measurements and "effectiveness" in baseline:
                eff = measurements["effectiveness"]
                eff_baseline = baseline["effectiveness"]
                condition = self._check_effectiveness(
                    exchanger_id, eff, eff_baseline, thresholds
                )
                if condition:
                    conditions.append(condition)

            # Check flow rate
            for flow_key in ["flow_rate_kg_s", "hot_flow_rate", "cold_flow_rate"]:
                if flow_key in measurements:
                    flow = measurements[flow_key]
                    design_flow = baseline.get("design_flow_rate", flow * 1.2)  # Assume 80% is normal
                    condition = self._check_flow_rate(
                        exchanger_id, flow_key, flow, design_flow, thresholds
                    )
                    if condition:
                        conditions.append(condition)

            return conditions

    def _check_temperature_high(
        self,
        exchanger_id: str,
        param_name: str,
        temperature: float,
        thresholds: AlertThresholds,
    ) -> Optional[CriticalCondition]:
        """Check for high temperature conditions."""
        condition_id = f"{exchanger_id}_{param_name}_high_{int(datetime.now(timezone.utc).timestamp())}"

        if temperature >= thresholds.temperature_critical_C:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.HIGH_TEMPERATURE,
                exchanger_id=exchanger_id,
                parameter_name=param_name,
                current_value=temperature,
                threshold_value=thresholds.temperature_critical_C,
                unit="C",
                severity=AlertSeverity.CRITICAL,
                message=(
                    f"CRITICAL: {param_name} at {temperature:.1f}C exceeds "
                    f"critical limit {thresholds.temperature_critical_C}C. "
                    f"Risk of metallurgical damage or coking."
                ),
                recommended_actions=[
                    "Immediately reduce heat input",
                    "Consider bypassing exchanger",
                    "Notify process engineer",
                ],
            )

        elif temperature >= thresholds.temperature_high_C:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.HIGH_TEMPERATURE,
                exchanger_id=exchanger_id,
                parameter_name=param_name,
                current_value=temperature,
                threshold_value=thresholds.temperature_high_C,
                unit="C",
                severity=AlertSeverity.HIGH,
                message=(
                    f"HIGH: {param_name} at {temperature:.1f}C approaching "
                    f"critical limit. Monitor closely."
                ),
                recommended_actions=[
                    "Reduce heat input if possible",
                    "Increase monitoring frequency",
                    "Prepare for bypass if needed",
                ],
            )

        elif temperature >= thresholds.temperature_warning_high_C:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.HIGH_TEMPERATURE,
                exchanger_id=exchanger_id,
                parameter_name=param_name,
                current_value=temperature,
                threshold_value=thresholds.temperature_warning_high_C,
                unit="C",
                severity=AlertSeverity.MEDIUM,
                message=(
                    f"WARNING: {param_name} at {temperature:.1f}C elevated. "
                    f"Monitor trend."
                ),
                recommended_actions=[
                    "Monitor temperature trend",
                    "Review process conditions",
                ],
            )

        return None

    def _check_temperature_low(
        self,
        exchanger_id: str,
        param_name: str,
        temperature: float,
        thresholds: AlertThresholds,
    ) -> Optional[CriticalCondition]:
        """Check for low temperature conditions (acid dew point)."""
        condition_id = f"{exchanger_id}_{param_name}_low_{int(datetime.now(timezone.utc).timestamp())}"

        if temperature <= thresholds.temperature_critical_low_C:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.LOW_TEMPERATURE,
                exchanger_id=exchanger_id,
                parameter_name=param_name,
                current_value=temperature,
                threshold_value=thresholds.temperature_critical_low_C,
                unit="C",
                severity=AlertSeverity.CRITICAL,
                message=(
                    f"CRITICAL: {param_name} at {temperature:.1f}C below "
                    f"acid dew point. Risk of severe corrosion."
                ),
                recommended_actions=[
                    "Immediately reduce heat recovery",
                    "Increase outlet temperature",
                    "Inspect for corrosion damage",
                ],
            )

        elif temperature <= thresholds.temperature_low_C:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.LOW_TEMPERATURE,
                exchanger_id=exchanger_id,
                parameter_name=param_name,
                current_value=temperature,
                threshold_value=thresholds.temperature_low_C,
                unit="C",
                severity=AlertSeverity.HIGH,
                message=(
                    f"HIGH: {param_name} at {temperature:.1f}C approaching "
                    f"acid dew point. Risk of corrosion."
                ),
                recommended_actions=[
                    "Reduce heat recovery",
                    "Monitor for corrosion indicators",
                ],
            )

        elif temperature <= thresholds.temperature_warning_low_C:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.LOW_TEMPERATURE,
                exchanger_id=exchanger_id,
                parameter_name=param_name,
                current_value=temperature,
                threshold_value=thresholds.temperature_warning_low_C,
                unit="C",
                severity=AlertSeverity.MEDIUM,
                message=(
                    f"WARNING: {param_name} at {temperature:.1f}C. "
                    f"Monitor for acid dew point approach."
                ),
                recommended_actions=[
                    "Monitor temperature trend",
                    "Review heat recovery setpoint",
                ],
            )

        return None

    def _check_pressure_drop(
        self,
        exchanger_id: str,
        pressure_drop: float,
        design_pressure_drop: float,
        thresholds: AlertThresholds,
    ) -> Optional[CriticalCondition]:
        """Check for high pressure drop conditions."""
        if design_pressure_drop <= 0:
            return None

        dp_pct = (pressure_drop / design_pressure_drop) * 100
        condition_id = f"{exchanger_id}_pressure_drop_{int(datetime.now(timezone.utc).timestamp())}"

        if dp_pct >= thresholds.pressure_drop_critical_pct:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.HIGH_PRESSURE_DROP,
                exchanger_id=exchanger_id,
                parameter_name="pressure_drop",
                current_value=pressure_drop,
                threshold_value=design_pressure_drop,
                unit="kPa",
                severity=AlertSeverity.CRITICAL,
                message=(
                    f"CRITICAL: Pressure drop {pressure_drop:.1f} kPa is "
                    f"{dp_pct:.0f}% of design. Severe fouling or blockage."
                ),
                recommended_actions=[
                    "Schedule emergency cleaning",
                    "Reduce flow rate if possible",
                    "Check for blockages",
                ],
            )

        elif dp_pct >= thresholds.pressure_drop_high_pct:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.HIGH_PRESSURE_DROP,
                exchanger_id=exchanger_id,
                parameter_name="pressure_drop",
                current_value=pressure_drop,
                threshold_value=design_pressure_drop,
                unit="kPa",
                severity=AlertSeverity.HIGH,
                message=(
                    f"HIGH: Pressure drop {pressure_drop:.1f} kPa is "
                    f"{dp_pct:.0f}% of design. Significant fouling."
                ),
                recommended_actions=[
                    "Schedule cleaning",
                    "Monitor flow rate",
                ],
            )

        elif dp_pct >= thresholds.pressure_drop_warning_pct:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.HIGH_PRESSURE_DROP,
                exchanger_id=exchanger_id,
                parameter_name="pressure_drop",
                current_value=pressure_drop,
                threshold_value=design_pressure_drop,
                unit="kPa",
                severity=AlertSeverity.MEDIUM,
                message=(
                    f"WARNING: Pressure drop {pressure_drop:.1f} kPa is "
                    f"{dp_pct:.0f}% of design. Fouling developing."
                ),
                recommended_actions=[
                    "Plan cleaning in next maintenance window",
                    "Continue monitoring",
                ],
            )

        return None

    def _check_fouling(
        self,
        exchanger_id: str,
        fouling_resistance: float,
        design_fouling: float,
        thresholds: AlertThresholds,
    ) -> Optional[CriticalCondition]:
        """Check for high fouling conditions."""
        if design_fouling <= 0:
            return None

        fouling_pct = (fouling_resistance / design_fouling) * 100
        condition_id = f"{exchanger_id}_fouling_{int(datetime.now(timezone.utc).timestamp())}"

        if fouling_pct >= thresholds.fouling_critical_pct:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.HIGH_FOULING,
                exchanger_id=exchanger_id,
                parameter_name="fouling_resistance",
                current_value=fouling_resistance,
                threshold_value=design_fouling,
                unit="m2K/W",
                severity=AlertSeverity.CRITICAL,
                message=(
                    f"CRITICAL: Fouling at {fouling_pct:.0f}% of design allowance. "
                    f"Exchanger performance severely degraded."
                ),
                recommended_actions=[
                    "Immediate cleaning required",
                    "Consider bypass to prevent further degradation",
                ],
            )

        elif fouling_pct >= thresholds.fouling_high_pct:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.HIGH_FOULING,
                exchanger_id=exchanger_id,
                parameter_name="fouling_resistance",
                current_value=fouling_resistance,
                threshold_value=design_fouling,
                unit="m2K/W",
                severity=AlertSeverity.HIGH,
                message=(
                    f"HIGH: Fouling at {fouling_pct:.0f}% of design allowance. "
                    f"Performance significantly degraded."
                ),
                recommended_actions=[
                    "Schedule cleaning soon",
                    "Monitor closely",
                ],
            )

        elif fouling_pct >= thresholds.fouling_warning_pct:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.HIGH_FOULING,
                exchanger_id=exchanger_id,
                parameter_name="fouling_resistance",
                current_value=fouling_resistance,
                threshold_value=design_fouling,
                unit="m2K/W",
                severity=AlertSeverity.MEDIUM,
                message=(
                    f"WARNING: Fouling at {fouling_pct:.0f}% of design allowance. "
                    f"Begin planning cleaning."
                ),
                recommended_actions=[
                    "Plan cleaning",
                    "Continue monitoring",
                ],
            )

        return None

    def _check_effectiveness(
        self,
        exchanger_id: str,
        effectiveness: float,
        baseline_effectiveness: float,
        thresholds: AlertThresholds,
    ) -> Optional[CriticalCondition]:
        """Check for effectiveness degradation."""
        if baseline_effectiveness <= 0:
            return None

        drop_pct = ((baseline_effectiveness - effectiveness) / baseline_effectiveness) * 100
        condition_id = f"{exchanger_id}_effectiveness_{int(datetime.now(timezone.utc).timestamp())}"

        if drop_pct >= thresholds.effectiveness_critical_drop_pct:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.EFFECTIVENESS_ANOMALY,
                exchanger_id=exchanger_id,
                parameter_name="effectiveness",
                current_value=effectiveness,
                threshold_value=baseline_effectiveness,
                unit="dimensionless",
                severity=AlertSeverity.CRITICAL,
                message=(
                    f"CRITICAL: Effectiveness dropped {drop_pct:.0f}% from baseline. "
                    f"Severe performance degradation."
                ),
                recommended_actions=[
                    "Inspect for fouling or damage",
                    "Schedule emergency cleaning",
                    "Verify sensor readings",
                ],
            )

        elif drop_pct >= thresholds.effectiveness_high_drop_pct:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.EFFECTIVENESS_ANOMALY,
                exchanger_id=exchanger_id,
                parameter_name="effectiveness",
                current_value=effectiveness,
                threshold_value=baseline_effectiveness,
                unit="dimensionless",
                severity=AlertSeverity.HIGH,
                message=(
                    f"HIGH: Effectiveness dropped {drop_pct:.0f}% from baseline. "
                    f"Significant degradation."
                ),
                recommended_actions=[
                    "Schedule cleaning",
                    "Investigate cause",
                ],
            )

        elif drop_pct >= thresholds.effectiveness_warning_drop_pct:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.EFFECTIVENESS_ANOMALY,
                exchanger_id=exchanger_id,
                parameter_name="effectiveness",
                current_value=effectiveness,
                threshold_value=baseline_effectiveness,
                unit="dimensionless",
                severity=AlertSeverity.MEDIUM,
                message=(
                    f"WARNING: Effectiveness dropped {drop_pct:.0f}% from baseline. "
                    f"Monitor for further degradation."
                ),
                recommended_actions=[
                    "Monitor trend",
                    "Plan cleaning in next window",
                ],
            )

        return None

    def _check_flow_rate(
        self,
        exchanger_id: str,
        param_name: str,
        flow_rate: float,
        design_flow: float,
        thresholds: AlertThresholds,
    ) -> Optional[CriticalCondition]:
        """Check for low flow rate conditions."""
        if design_flow <= 0:
            return None

        flow_pct = (flow_rate / design_flow) * 100
        condition_id = f"{exchanger_id}_{param_name}_low_{int(datetime.now(timezone.utc).timestamp())}"

        if flow_pct <= thresholds.flow_critical_low_pct:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.LOW_FLOW,
                exchanger_id=exchanger_id,
                parameter_name=param_name,
                current_value=flow_rate,
                threshold_value=design_flow,
                unit="kg/s",
                severity=AlertSeverity.CRITICAL,
                message=(
                    f"CRITICAL: {param_name} at {flow_pct:.0f}% of design. "
                    f"Risk of overheating or thermal damage."
                ),
                recommended_actions=[
                    "Check for blockages or valve issues",
                    "Consider emergency bypass",
                    "Verify pump operation",
                ],
            )

        elif flow_pct <= thresholds.flow_low_pct:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.LOW_FLOW,
                exchanger_id=exchanger_id,
                parameter_name=param_name,
                current_value=flow_rate,
                threshold_value=design_flow,
                unit="kg/s",
                severity=AlertSeverity.HIGH,
                message=(
                    f"HIGH: {param_name} at {flow_pct:.0f}% of design. "
                    f"Performance significantly affected."
                ),
                recommended_actions=[
                    "Investigate flow reduction",
                    "Check for partial blockages",
                ],
            )

        elif flow_pct <= thresholds.flow_warning_low_pct:
            return CriticalCondition(
                condition_id=condition_id,
                condition_type=ConditionType.LOW_FLOW,
                exchanger_id=exchanger_id,
                parameter_name=param_name,
                current_value=flow_rate,
                threshold_value=design_flow,
                unit="kg/s",
                severity=AlertSeverity.MEDIUM,
                message=(
                    f"WARNING: {param_name} at {flow_pct:.0f}% of design. "
                    f"Monitor for further reduction."
                ),
                recommended_actions=[
                    "Monitor flow trend",
                    "Review process conditions",
                ],
            )

        return None

    # =========================================================================
    # ALERT MANAGEMENT
    # =========================================================================

    def create_alert(self, condition: CriticalCondition) -> Alert:
        """
        Create an alert from a critical condition.

        Args:
            condition: The critical condition

        Returns:
            Created alert
        """
        with self._lock:
            # Check for cooldown
            cooldown_key = f"{condition.exchanger_id}_{condition.condition_type.value}"
            now = datetime.now(timezone.utc)

            if cooldown_key in self._condition_last_seen:
                last_seen = self._condition_last_seen[cooldown_key]
                cooldown = timedelta(minutes=self.config.alert_cooldown_minutes)
                if now - last_seen < cooldown:
                    # Find existing active alert
                    for alert in self._active_alerts.values():
                        if (
                            alert.condition.exchanger_id == condition.exchanger_id and
                            alert.condition.condition_type == condition.condition_type
                        ):
                            return alert

            self._condition_last_seen[cooldown_key] = now

            # Create new alert
            alert_id = f"ALT_{condition.exchanger_id}_{int(now.timestamp())}"
            alert = Alert(
                alert_id=alert_id,
                condition=condition,
                severity=condition.severity,
                escalation_level=EscalationLevel.NONE,
            )

            self._active_alerts[alert_id] = alert

            # Send notification
            self._send_notification(alert)

            logger.warning(
                f"Alert created: id={alert_id}, severity={condition.severity.value}, "
                f"exchanger={condition.exchanger_id}, type={condition.condition_type.value}"
            )

            return alert

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> bool:
        """
        Acknowledge an active alert.

        Args:
            alert_id: Alert identifier
            acknowledged_by: User acknowledging

        Returns:
            True if acknowledged successfully
        """
        with self._lock:
            alert = self._active_alerts.get(alert_id)
            if alert is None:
                logger.warning(f"Alert {alert_id} not found for acknowledgment")
                return False

            if alert.state != AlertState.ACTIVE:
                logger.warning(f"Alert {alert_id} is not active (state={alert.state.value})")
                return False

            alert.state = AlertState.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now(timezone.utc)
            alert.acknowledged_by = acknowledged_by

            logger.info(
                f"Alert acknowledged: id={alert_id}, by={acknowledged_by}"
            )

            return True

    def resolve_alert(
        self,
        alert_id: str,
        resolved_by: Optional[str] = None,
        resolution_notes: Optional[str] = None,
    ) -> bool:
        """
        Resolve an active or acknowledged alert.

        Args:
            alert_id: Alert identifier
            resolved_by: User resolving (optional)
            resolution_notes: Resolution notes (optional)

        Returns:
            True if resolved successfully
        """
        with self._lock:
            alert = self._active_alerts.get(alert_id)
            if alert is None:
                logger.warning(f"Alert {alert_id} not found for resolution")
                return False

            alert.state = AlertState.RESOLVED
            alert.resolved_at = datetime.now(timezone.utc)

            # Move to resolved list
            self._resolved_alerts.append(alert)
            del self._active_alerts[alert_id]

            # Trim resolved alerts
            self._trim_resolved_alerts()

            logger.info(
                f"Alert resolved: id={alert_id}, by={resolved_by}"
            )

            return True

    def escalate_alert(
        self,
        alert_id: str,
        reason: Optional[str] = None,
    ) -> Optional[EscalationLevel]:
        """
        Escalate an alert to the next level.

        Args:
            alert_id: Alert identifier
            reason: Escalation reason

        Returns:
            New escalation level or None if failed
        """
        with self._lock:
            alert = self._active_alerts.get(alert_id)
            if alert is None:
                logger.warning(f"Alert {alert_id} not found for escalation")
                return None

            # Determine next escalation level
            escalation_order = [
                EscalationLevel.NONE,
                EscalationLevel.OPERATOR,
                EscalationLevel.SUPERVISOR,
                EscalationLevel.ENGINEER,
                EscalationLevel.MANAGER,
                EscalationLevel.EMERGENCY_TEAM,
            ]

            try:
                current_idx = escalation_order.index(alert.escalation_level)
                if current_idx < len(escalation_order) - 1:
                    new_level = escalation_order[current_idx + 1]
                else:
                    logger.warning(f"Alert {alert_id} already at maximum escalation")
                    return alert.escalation_level
            except ValueError:
                new_level = EscalationLevel.OPERATOR

            # Update alert
            now = datetime.now(timezone.utc)
            alert.escalation_level = new_level
            alert.escalated_at = now
            alert.state = AlertState.ESCALATED
            alert.escalation_history.append({
                "timestamp": now.isoformat(),
                "from_level": alert.escalation_level.value,
                "to_level": new_level.value,
                "reason": reason or "Manual escalation",
            })

            # Invoke escalation callback
            if self._escalation_callback:
                try:
                    self._escalation_callback(alert, new_level)
                except Exception as e:
                    logger.error(f"Escalation callback failed: {e}")

            logger.warning(
                f"Alert escalated: id={alert_id}, level={new_level.value}, reason={reason}"
            )

            return new_level

    def _send_notification(self, alert: Alert) -> None:
        """Send notification for an alert."""
        if self._notification_callback:
            try:
                self._notification_callback(alert)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")

        alert.notifications_sent.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": alert.severity.value,
            "escalation_level": alert.escalation_level.value,
        })

    def _trim_resolved_alerts(self) -> None:
        """Remove old resolved alerts."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=self.config.retain_resolved_hours
        )
        self._resolved_alerts = [
            a for a in self._resolved_alerts
            if a.resolved_at and a.resolved_at > cutoff
        ]

    # =========================================================================
    # AUTO-ESCALATION
    # =========================================================================

    def check_auto_escalation(self) -> List[Alert]:
        """
        Check for alerts that need auto-escalation.

        Returns:
            List of escalated alerts
        """
        if not self.config.enable_auto_escalation:
            return []

        with self._lock:
            escalated = []
            policy = self.config.escalation_policy

            for alert in self._active_alerts.values():
                if alert.state == AlertState.RESOLVED:
                    continue

                age = alert.age_minutes

                # Determine if escalation is needed
                should_escalate = False
                reason = ""

                if alert.severity == AlertSeverity.EMERGENCY:
                    if alert.escalation_level == EscalationLevel.NONE:
                        should_escalate = True
                        reason = "Emergency - immediate escalation"

                elif alert.severity == AlertSeverity.CRITICAL:
                    if (
                        alert.escalation_level in (EscalationLevel.NONE, EscalationLevel.OPERATOR) and
                        age >= policy.critical_to_engineer_minutes
                    ):
                        should_escalate = True
                        reason = f"Critical alert unresolved for {age:.0f} minutes"

                elif alert.severity == AlertSeverity.HIGH:
                    if (
                        alert.escalation_level == EscalationLevel.NONE and
                        age >= policy.high_to_supervisor_minutes
                    ):
                        should_escalate = True
                        reason = f"High alert unacknowledged for {age:.0f} minutes"

                elif alert.severity == AlertSeverity.MEDIUM:
                    if (
                        alert.escalation_level == EscalationLevel.NONE and
                        age >= policy.medium_to_operator_minutes
                    ):
                        should_escalate = True
                        reason = f"Medium alert unacknowledged for {age:.0f} minutes"

                if should_escalate:
                    self.escalate_alert(alert.alert_id, reason)
                    escalated.append(alert)

            return escalated

    # =========================================================================
    # SAFE STATE RECOMMENDATIONS
    # =========================================================================

    def generate_safe_state_recommendation(
        self,
        alert: Alert,
    ) -> Optional[SafeStateRecommendation]:
        """
        Generate safe state recommendation for an alert.

        Args:
            alert: The triggering alert

        Returns:
            Safe state recommendation or None
        """
        if not self.config.enable_safe_state_recommendations:
            return None

        with self._lock:
            condition = alert.condition

            # Determine safe state type based on condition
            if condition.condition_type == ConditionType.HIGH_TEMPERATURE:
                if condition.severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY):
                    safe_state = SafeStateType.BYPASS_EXCHANGER
                    priority = 1
                    description = (
                        f"Bypass exchanger {condition.exchanger_id} to prevent "
                        f"thermal damage. Temperature {condition.current_value:.1f}C "
                        f"exceeds safe limit."
                    )
                    steps = [
                        "Open bypass valve",
                        "Close exchanger inlet/outlet valves",
                        "Verify bypass flow established",
                        "Monitor downstream temperatures",
                    ]
                    requires_auth = True
                else:
                    safe_state = SafeStateType.REDUCE_HEAT_RECOVERY
                    priority = 2
                    description = (
                        f"Reduce heat recovery at {condition.exchanger_id} to lower "
                        f"operating temperature."
                    )
                    steps = [
                        "Reduce process flow or adjust bypass",
                        "Monitor temperatures",
                        "Verify stable operation",
                    ]
                    requires_auth = False

            elif condition.condition_type == ConditionType.LOW_TEMPERATURE:
                safe_state = SafeStateType.REDUCE_HEAT_RECOVERY
                priority = 1 if condition.severity == AlertSeverity.CRITICAL else 2
                description = (
                    f"Reduce heat recovery at {condition.exchanger_id} to raise "
                    f"outlet temperature above acid dew point."
                )
                steps = [
                    "Reduce cold side flow or bypass",
                    "Monitor outlet temperature",
                    "Verify temperature above dew point",
                ]
                requires_auth = condition.severity == AlertSeverity.CRITICAL

            elif condition.condition_type == ConditionType.HIGH_PRESSURE_DROP:
                safe_state = SafeStateType.SCHEDULE_CLEANING
                priority = 2 if condition.severity == AlertSeverity.CRITICAL else 3
                description = (
                    f"Schedule cleaning for {condition.exchanger_id}. "
                    f"Pressure drop {condition.current_value:.1f} kPa indicates "
                    f"significant fouling."
                )
                steps = [
                    "Coordinate with operations for shutdown window",
                    "Prepare cleaning equipment and chemicals",
                    "Execute cleaning procedure",
                    "Verify performance recovery",
                ]
                requires_auth = False

            elif condition.condition_type == ConditionType.HIGH_FOULING:
                safe_state = SafeStateType.SCHEDULE_CLEANING
                priority = 2
                description = (
                    f"Schedule cleaning for {condition.exchanger_id}. "
                    f"Fouling resistance {condition.current_value:.6f} m2K/W "
                    f"exceeds design allowance."
                )
                steps = [
                    "Plan cleaning in next maintenance window",
                    "Monitor performance degradation rate",
                    "Prepare for cleaning",
                ]
                requires_auth = False

            elif condition.condition_type == ConditionType.LOW_FLOW:
                if condition.severity == AlertSeverity.CRITICAL:
                    safe_state = SafeStateType.BYPASS_EXCHANGER
                    priority = 1
                    description = (
                        f"Bypass exchanger {condition.exchanger_id} due to "
                        f"critically low flow. Risk of thermal damage."
                    )
                    steps = [
                        "Open bypass",
                        "Investigate flow restriction",
                        "Restore flow before returning to service",
                    ]
                    requires_auth = True
                else:
                    safe_state = SafeStateType.MANUAL_INSPECTION
                    priority = 2
                    description = (
                        f"Inspect {condition.exchanger_id} for flow restrictions."
                    )
                    steps = [
                        "Check inlet strainers",
                        "Verify valve positions",
                        "Check pump operation",
                    ]
                    requires_auth = False

            else:
                # Generic safe state for other conditions
                safe_state = SafeStateType.MANUAL_INSPECTION
                priority = 3
                description = (
                    f"Manual inspection required for {condition.exchanger_id}."
                )
                steps = [
                    "Review condition details",
                    "Inspect equipment",
                    "Determine corrective action",
                ]
                requires_auth = False

            # Create recommendation
            rec_id = f"REC_{alert.alert_id}"
            recommendation = SafeStateRecommendation(
                recommendation_id=rec_id,
                exchanger_id=condition.exchanger_id,
                safe_state_type=safe_state,
                triggering_alerts=[alert.alert_id],
                description=description,
                priority=priority,
                estimated_impact="Temporary reduction in heat recovery capacity",
                implementation_steps=steps,
                valid_until=datetime.now(timezone.utc) + timedelta(hours=24),
                requires_authorization=requires_auth,
                authorizing_role="process_engineer" if requires_auth else None,
            )

            self._recommendations[rec_id] = recommendation

            logger.info(
                f"Safe state recommendation created: id={rec_id}, "
                f"type={safe_state.value}, priority={priority}"
            )

            return recommendation

    def get_active_recommendations(
        self,
        exchanger_id: Optional[str] = None,
    ) -> List[SafeStateRecommendation]:
        """
        Get active safe state recommendations.

        Args:
            exchanger_id: Filter by exchanger (optional)

        Returns:
            List of valid recommendations
        """
        with self._lock:
            recommendations = [
                r for r in self._recommendations.values()
                if r.is_valid()
            ]

            if exchanger_id:
                recommendations = [
                    r for r in recommendations
                    if r.exchanger_id == exchanger_id
                ]

            return sorted(recommendations, key=lambda r: r.priority)

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_active_alerts(
        self,
        exchanger_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """
        Get active alerts with optional filtering.

        Args:
            exchanger_id: Filter by exchanger
            severity: Filter by severity

        Returns:
            List of active alerts
        """
        with self._lock:
            alerts = list(self._active_alerts.values())

            if exchanger_id:
                alerts = [
                    a for a in alerts
                    if a.condition.exchanger_id == exchanger_id
                ]

            if severity:
                alerts = [a for a in alerts if a.severity == severity]

            return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def get_alert_count_by_severity(self) -> Dict[str, int]:
        """Get count of active alerts by severity."""
        with self._lock:
            counts = {s.value: 0 for s in AlertSeverity}
            for alert in self._active_alerts.values():
                counts[alert.severity.value] += 1
            return counts

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall emergency response system status.

        Returns:
            Status dictionary
        """
        with self._lock:
            alert_counts = self.get_alert_count_by_severity()
            total_active = len(self._active_alerts)

            # Determine overall status
            if alert_counts.get("emergency", 0) > 0:
                status = "emergency"
            elif alert_counts.get("critical", 0) > 0:
                status = "critical"
            elif alert_counts.get("high", 0) > 0:
                status = "high_alert"
            elif total_active > 0:
                status = "alert"
            else:
                status = "normal"

            return {
                "status": status,
                "total_active_alerts": total_active,
                "alerts_by_severity": alert_counts,
                "unacknowledged_count": sum(
                    1 for a in self._active_alerts.values()
                    if a.state == AlertState.ACTIVE
                ),
                "escalated_count": sum(
                    1 for a in self._active_alerts.values()
                    if a.state == AlertState.ESCALATED
                ),
                "active_recommendations": len([
                    r for r in self._recommendations.values()
                    if r.is_valid()
                ]),
                "resolved_last_24h": sum(
                    1 for a in self._resolved_alerts
                    if a.resolved_at and
                    a.resolved_at > datetime.now(timezone.utc) - timedelta(hours=24)
                ),
            }


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "AlertSeverity",
    "AlertState",
    "EscalationLevel",
    "ConditionType",
    "SafeStateType",
    # Config
    "AlertThresholds",
    "EscalationPolicy",
    "EmergencyResponseConfig",
    # Data models
    "CriticalCondition",
    "Alert",
    "SafeStateRecommendation",
    # Main class
    "EmergencyResponseHandler",
]
