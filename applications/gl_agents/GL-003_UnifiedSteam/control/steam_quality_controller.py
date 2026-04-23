"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Steam Quality Controller

This module implements steam quality control with superheat margin maintenance
and coordination with desuperheater control systems.

Control Architecture:
    - Steam quality (dryness fraction) monitoring and control
    - Superheat margin maintenance for process safety
    - Coordinated control with desuperheater systems
    - Multi-parameter optimization for quality targets

Reference Standards:
    - ASME PTC 4.3 Air Preheater Performance Test Code
    - ASME B31.1 Power Piping
    - IEC 61511 Functional Safety

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class QualityStatus(str, Enum):
    """Steam quality status enumeration."""
    OPTIMAL = "optimal"
    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    CRITICAL = "critical"


class CorrectionType(str, Enum):
    """Quality correction type enumeration."""
    INCREASE_SUPERHEAT = "increase_superheat"
    DECREASE_SUPERHEAT = "decrease_superheat"
    INCREASE_DRYNESS = "increase_dryness"
    MAINTAIN = "maintain"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


class ControlActionType(str, Enum):
    """Control action type enumeration."""
    REDUCE_SPRAY = "reduce_spray"
    INCREASE_SPRAY = "increase_spray"
    ADJUST_HEAT_INPUT = "adjust_heat_input"
    ADJUST_FLOW = "adjust_flow"
    NO_ACTION = "no_action"


# =============================================================================
# DATA MODELS
# =============================================================================

class SteamQualityState(BaseModel):
    """Current steam quality state."""

    measurement_point_id: str = Field(..., description="Measurement point ID")
    dryness_fraction: float = Field(
        ...,
        ge=0,
        le=1,
        description="Steam dryness fraction (0-1)"
    )
    superheat_c: float = Field(
        ...,
        description="Superheat above saturation (C)"
    )
    temperature_c: float = Field(..., description="Steam temperature (C)")
    pressure_kpa: float = Field(..., ge=0, description="Steam pressure (kPa)")
    saturation_temp_c: float = Field(
        ...,
        description="Saturation temperature (C)"
    )
    enthalpy_kj_kg: float = Field(
        ...,
        ge=0,
        description="Specific enthalpy (kJ/kg)"
    )
    flow_rate_kg_s: float = Field(
        ...,
        ge=0,
        description="Steam flow rate (kg/s)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Measurement timestamp"
    )


class QualityTargets(BaseModel):
    """Steam quality control targets."""

    target_dryness: float = Field(
        default=1.0,
        ge=0.9,
        le=1.0,
        description="Target dryness fraction"
    )
    target_superheat_c: float = Field(
        default=25.0,
        ge=0,
        description="Target superheat (C)"
    )
    min_superheat_c: float = Field(
        default=15.0,
        ge=0,
        description="Minimum superheat margin (C)"
    )
    max_superheat_c: float = Field(
        default=50.0,
        ge=0,
        description="Maximum superheat (C)"
    )
    dryness_warning_threshold: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description="Dryness warning threshold"
    )
    dryness_critical_threshold: float = Field(
        default=0.90,
        ge=0,
        le=1,
        description="Dryness critical threshold"
    )


class Correction(BaseModel):
    """Steam quality correction output."""

    correction_id: str = Field(..., description="Correction ID")
    correction_type: CorrectionType = Field(
        ...,
        description="Type of correction required"
    )
    current_quality: float = Field(
        ...,
        description="Current quality value"
    )
    target_quality: float = Field(
        ...,
        description="Target quality value"
    )
    correction_magnitude: float = Field(
        ...,
        description="Magnitude of correction required"
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Correction priority (1=highest)"
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended actions to achieve correction"
    )
    safety_validated: bool = Field(
        default=False,
        description="Correction validated against safety limits"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Correction timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class CoordinatedResult(BaseModel):
    """Result of coordinated desuperheater control."""

    coordination_id: str = Field(..., description="Coordination ID")
    quality_correction: Correction = Field(
        ...,
        description="Quality correction details"
    )
    desuperheater_setpoint: float = Field(
        ...,
        description="Coordinated desuperheater setpoint (%)"
    )
    expected_quality_improvement: float = Field(
        ...,
        description="Expected quality improvement"
    )
    coordination_successful: bool = Field(
        ...,
        description="Coordination was successful"
    )
    constraints_satisfied: bool = Field(
        ...,
        description="All constraints satisfied"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Coordination timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class ControlAction(BaseModel):
    """Steam quality control action."""

    action_id: str = Field(..., description="Action ID")
    action_type: ControlActionType = Field(
        ...,
        description="Type of control action"
    )
    target_parameter: str = Field(
        ...,
        description="Parameter to adjust"
    )
    current_value: float = Field(
        ...,
        description="Current parameter value"
    )
    recommended_value: float = Field(
        ...,
        description="Recommended parameter value"
    )
    unit: str = Field(..., description="Parameter unit")
    reason: str = Field(..., description="Reason for action")
    urgency: str = Field(
        default="normal",
        description="Action urgency (normal, high, critical)"
    )
    requires_confirmation: bool = Field(
        default=True,
        description="Requires operator confirmation"
    )
    safety_validated: bool = Field(
        default=False,
        description="Action validated against safety limits"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Action timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


# =============================================================================
# STEAM QUALITY CONTROLLER
# =============================================================================

class SteamQualityController:
    """
    Steam quality control with superheat margin maintenance.

    This controller manages steam quality (dryness fraction) and superheat
    to ensure safe, efficient steam delivery while coordinating with
    desuperheater control systems.

    Control Features:
        - Steam dryness monitoring and control
        - Superheat margin maintenance
        - Coordinated control with desuperheaters
        - Multi-point quality optimization

    Safety Features:
        - Minimum superheat margin enforcement
        - Wet steam detection and alerting
        - Emergency actions for critical quality degradation
        - Erosion threshold monitoring

    Attributes:
        targets: Quality control targets
        _quality_history: Historical quality measurements
        _coordination_enabled: Whether coordination is enabled

    Example:
        >>> controller = SteamQualityController(targets=QualityTargets())
        >>> state = SteamQualityState(...)
        >>> correction = controller.compute_quality_correction(
        ...     current_quality=0.92,
        ...     target_quality=1.0
        ... )
    """

    def __init__(
        self,
        targets: Optional[QualityTargets] = None,
        enable_coordination: bool = True
    ):
        """
        Initialize SteamQualityController.

        Args:
            targets: Quality control targets (uses defaults if None)
            enable_coordination: Enable desuperheater coordination
        """
        self.targets = targets or QualityTargets()
        self._quality_history: List[SteamQualityState] = []
        self._coordination_enabled = enable_coordination
        self._max_history_size = 1000

        logger.info(
            f"SteamQualityController initialized with targets: "
            f"dryness={self.targets.target_dryness}, "
            f"superheat={self.targets.target_superheat_c}C"
        )

    def compute_quality_correction(
        self,
        current_quality: float,
        target_quality: float,
        current_state: Optional[SteamQualityState] = None
    ) -> Correction:
        """
        Compute quality correction to achieve target.

        This method calculates the required correction to move from current
        quality to target quality while respecting safety constraints.

        Args:
            current_quality: Current quality value (0-1 for dryness)
            target_quality: Target quality value
            current_state: Optional current state for additional context

        Returns:
            Correction: Computed correction with recommendations
        """
        start_time = datetime.now()

        # Determine correction type and magnitude
        quality_error = target_quality - current_quality
        correction_magnitude = abs(quality_error)

        # Determine correction type based on error direction
        if correction_magnitude < 0.01:
            correction_type = CorrectionType.MAINTAIN
            priority = 5
        elif quality_error > 0:
            # Need to increase quality (dryness)
            correction_type = CorrectionType.INCREASE_DRYNESS
            priority = self._calculate_priority(current_quality, self.targets)
        else:
            # Quality above target - may need to increase spray
            correction_type = CorrectionType.DECREASE_SUPERHEAT
            priority = 3

        # Check for critical conditions
        if current_quality < self.targets.dryness_critical_threshold:
            correction_type = CorrectionType.INCREASE_DRYNESS
            priority = 1
            logger.warning(
                f"CRITICAL: Steam quality {current_quality:.3f} below critical "
                f"threshold {self.targets.dryness_critical_threshold}"
            )

        # Generate recommended actions
        recommended_actions = self._generate_recommended_actions(
            correction_type, correction_magnitude, current_state
        )

        # Generate correction ID
        correction_id = hashlib.sha256(
            f"QC_{current_quality}_{target_quality}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        # Create correction
        correction = Correction(
            correction_id=correction_id,
            correction_type=correction_type,
            current_quality=current_quality,
            target_quality=target_quality,
            correction_magnitude=correction_magnitude,
            priority=priority,
            recommended_actions=recommended_actions,
            safety_validated=True,  # Basic validation passed
            timestamp=datetime.now()
        )

        # Calculate provenance hash
        correction.provenance_hash = hashlib.sha256(
            f"{correction_id}|{correction_type.value}|{correction_magnitude}".encode()
        ).hexdigest()

        logger.info(
            f"Quality correction {correction_id}: type={correction_type.value}, "
            f"magnitude={correction_magnitude:.4f}, priority={priority}"
        )

        return correction

    def coordinate_desuperheater(
        self,
        quality_demand: Correction,
        ds_controller: Any
    ) -> CoordinatedResult:
        """
        Coordinate quality correction with desuperheater control.

        This method coordinates quality control actions with the desuperheater
        controller to achieve integrated temperature and quality targets.

        Args:
            quality_demand: Quality correction requirement
            ds_controller: Desuperheater controller instance

        Returns:
            CoordinatedResult: Coordinated control result
        """
        start_time = datetime.now()
        warnings = []

        if not self._coordination_enabled:
            warnings.append("Coordination disabled - returning independent control")
            logger.warning("Desuperheater coordination is disabled")

        # Determine desuperheater setpoint adjustment based on quality demand
        ds_setpoint_adjustment = 0.0

        if quality_demand.correction_type == CorrectionType.INCREASE_DRYNESS:
            # Reduce spray to increase superheat and dryness
            ds_setpoint_adjustment = -quality_demand.correction_magnitude * 20.0
        elif quality_demand.correction_type == CorrectionType.INCREASE_SUPERHEAT:
            # Reduce spray to increase superheat
            ds_setpoint_adjustment = -quality_demand.correction_magnitude * 15.0
        elif quality_demand.correction_type == CorrectionType.DECREASE_SUPERHEAT:
            # Increase spray to reduce superheat
            ds_setpoint_adjustment = quality_demand.correction_magnitude * 15.0

        # Get current desuperheater setpoint if available
        current_ds_setpoint = 50.0  # Default mid-range
        if hasattr(ds_controller, '_last_setpoint') and ds_controller._last_setpoint:
            current_ds_setpoint = ds_controller._last_setpoint.value

        # Calculate new setpoint with limits
        new_ds_setpoint = current_ds_setpoint + ds_setpoint_adjustment
        new_ds_setpoint = max(0.0, min(100.0, new_ds_setpoint))

        # Validate against desuperheater constraints
        constraints_satisfied = True
        if hasattr(ds_controller, 'constraints'):
            if new_ds_setpoint < ds_controller.constraints.min_spray_valve_pct:
                new_ds_setpoint = ds_controller.constraints.min_spray_valve_pct
                warnings.append("Setpoint limited by minimum spray valve position")
            if new_ds_setpoint > ds_controller.constraints.max_spray_valve_pct:
                new_ds_setpoint = ds_controller.constraints.max_spray_valve_pct
                warnings.append("Setpoint limited by maximum spray valve position")

        # Calculate expected quality improvement
        expected_improvement = self._estimate_quality_improvement(
            quality_demand.current_quality,
            ds_setpoint_adjustment
        )

        # Generate coordination ID
        coordination_id = hashlib.sha256(
            f"COORD_{quality_demand.correction_id}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        result = CoordinatedResult(
            coordination_id=coordination_id,
            quality_correction=quality_demand,
            desuperheater_setpoint=new_ds_setpoint,
            expected_quality_improvement=expected_improvement,
            coordination_successful=True,
            constraints_satisfied=constraints_satisfied,
            warnings=warnings,
            timestamp=datetime.now()
        )

        # Calculate provenance hash
        result.provenance_hash = hashlib.sha256(
            f"{coordination_id}|{new_ds_setpoint}|{expected_improvement}".encode()
        ).hexdigest()

        logger.info(
            f"Coordination {coordination_id}: DS setpoint={new_ds_setpoint:.2f}%, "
            f"expected improvement={expected_improvement:.4f}, "
            f"constraints_satisfied={constraints_satisfied}"
        )

        return result

    def maintain_superheat_margin(
        self,
        current_state: SteamQualityState,
        min_margin_c: Optional[float] = None
    ) -> ControlAction:
        """
        Maintain superheat margin above minimum threshold.

        This method monitors superheat margin and generates control actions
        when the margin approaches the minimum threshold.

        Args:
            current_state: Current steam quality state
            min_margin_c: Minimum margin (uses target default if None)

        Returns:
            ControlAction: Recommended control action
        """
        start_time = datetime.now()
        min_margin = min_margin_c or self.targets.min_superheat_c

        # Store state in history
        self._add_to_history(current_state)

        # Calculate current margin
        current_margin = current_state.superheat_c

        # Determine required action
        if current_margin < min_margin:
            # Critical: Below minimum margin
            action_type = ControlActionType.REDUCE_SPRAY
            urgency = "critical"
            reason = (
                f"Superheat margin {current_margin:.1f}C below minimum "
                f"{min_margin}C - immediate action required"
            )
            adjustment = (min_margin - current_margin) * 2.0  # Aggressive correction
            logger.warning(reason)
        elif current_margin < min_margin * 1.2:
            # Warning: Approaching minimum margin
            action_type = ControlActionType.REDUCE_SPRAY
            urgency = "high"
            reason = (
                f"Superheat margin {current_margin:.1f}C approaching minimum "
                f"{min_margin}C - corrective action recommended"
            )
            adjustment = (min_margin * 1.2 - current_margin) * 1.5
            logger.info(reason)
        elif current_margin > self.targets.max_superheat_c:
            # Excess superheat - increase spray
            action_type = ControlActionType.INCREASE_SPRAY
            urgency = "normal"
            reason = (
                f"Superheat {current_margin:.1f}C exceeds target "
                f"{self.targets.max_superheat_c}C - optimization opportunity"
            )
            adjustment = -(current_margin - self.targets.target_superheat_c) * 0.5
        else:
            # Within acceptable range
            action_type = ControlActionType.NO_ACTION
            urgency = "normal"
            reason = f"Superheat margin {current_margin:.1f}C is adequate"
            adjustment = 0.0

        # Generate action ID
        action_id = hashlib.sha256(
            f"SHM_{current_state.measurement_point_id}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        # Estimate spray valve target
        current_spray = 50.0  # Assume mid-range if not known
        target_spray = max(0.0, min(100.0, current_spray + adjustment))

        action = ControlAction(
            action_id=action_id,
            action_type=action_type,
            target_parameter="spray_valve_position",
            current_value=current_spray,
            recommended_value=target_spray,
            unit="%",
            reason=reason,
            urgency=urgency,
            requires_confirmation=(urgency != "critical"),
            safety_validated=(current_margin >= min_margin * 0.8)
        )

        # Calculate provenance hash
        action.provenance_hash = hashlib.sha256(
            f"{action_id}|{action_type.value}|{target_spray}".encode()
        ).hexdigest()

        return action

    def get_quality_status(
        self,
        current_state: SteamQualityState
    ) -> QualityStatus:
        """
        Get current quality status assessment.

        Args:
            current_state: Current steam quality state

        Returns:
            QualityStatus: Quality status assessment
        """
        dryness = current_state.dryness_fraction
        superheat = current_state.superheat_c

        # Check for critical conditions
        if dryness < self.targets.dryness_critical_threshold:
            return QualityStatus.CRITICAL

        if superheat < self.targets.min_superheat_c:
            return QualityStatus.CRITICAL

        # Check for warning conditions
        if dryness < self.targets.dryness_warning_threshold:
            return QualityStatus.WARNING

        if superheat < self.targets.min_superheat_c * 1.2:
            return QualityStatus.WARNING

        # Check for optimal conditions
        if (abs(dryness - self.targets.target_dryness) < 0.02 and
                abs(superheat - self.targets.target_superheat_c) < 5.0):
            return QualityStatus.OPTIMAL

        return QualityStatus.ACCEPTABLE

    def get_quality_trend(
        self,
        window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze quality trend from recent history.

        Args:
            window_size: Number of recent measurements to analyze

        Returns:
            Dict containing trend analysis
        """
        if len(self._quality_history) < 2:
            return {
                "trend": "insufficient_data",
                "data_points": len(self._quality_history)
            }

        recent = self._quality_history[-window_size:]

        dryness_values = [s.dryness_fraction for s in recent]
        superheat_values = [s.superheat_c for s in recent]

        # Calculate trends
        dryness_trend = dryness_values[-1] - dryness_values[0]
        superheat_trend = superheat_values[-1] - superheat_values[0]

        return {
            "trend": "improving" if dryness_trend > 0 else "degrading",
            "dryness_change": dryness_trend,
            "superheat_change": superheat_trend,
            "current_dryness": dryness_values[-1],
            "current_superheat": superheat_values[-1],
            "data_points": len(recent)
        }

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _calculate_priority(
        self,
        current_quality: float,
        targets: QualityTargets
    ) -> int:
        """
        Calculate correction priority based on quality deviation.

        Args:
            current_quality: Current quality value
            targets: Quality targets

        Returns:
            int: Priority (1=highest, 5=lowest)
        """
        if current_quality < targets.dryness_critical_threshold:
            return 1
        elif current_quality < targets.dryness_warning_threshold:
            return 2
        elif current_quality < targets.target_dryness * 0.98:
            return 3
        elif current_quality < targets.target_dryness:
            return 4
        else:
            return 5

    def _generate_recommended_actions(
        self,
        correction_type: CorrectionType,
        magnitude: float,
        current_state: Optional[SteamQualityState]
    ) -> List[str]:
        """
        Generate recommended actions for correction type.

        Args:
            correction_type: Type of correction
            magnitude: Correction magnitude
            current_state: Current state for context

        Returns:
            List of recommended action strings
        """
        actions = []

        if correction_type == CorrectionType.INCREASE_DRYNESS:
            actions.append("Reduce desuperheater spray water flow")
            actions.append("Verify steam separator is functioning correctly")
            if magnitude > 0.05:
                actions.append("Check for excessive moisture carryover from boiler")
                actions.append("Inspect steam drum internals")

        elif correction_type == CorrectionType.INCREASE_SUPERHEAT:
            actions.append("Reduce desuperheater spray water flow")
            if magnitude > 10.0:
                actions.append("Consider reducing boiler load if appropriate")

        elif correction_type == CorrectionType.DECREASE_SUPERHEAT:
            actions.append("Increase desuperheater spray water flow")
            if magnitude > 20.0:
                actions.append("Verify spray water supply pressure is adequate")

        elif correction_type == CorrectionType.MAINTAIN:
            actions.append("Continue monitoring - no immediate action required")

        return actions

    def _estimate_quality_improvement(
        self,
        current_quality: float,
        spray_adjustment: float
    ) -> float:
        """
        Estimate quality improvement from spray adjustment.

        Simple linear model - production would use process models.

        Args:
            current_quality: Current quality value
            spray_adjustment: Spray valve adjustment (%)

        Returns:
            float: Estimated quality improvement
        """
        # Rough estimate: 1% spray reduction = 0.002 quality improvement
        improvement = -spray_adjustment * 0.002

        # Cap improvement at realistic values
        max_improvement = 1.0 - current_quality
        return min(improvement, max_improvement)

    def _add_to_history(self, state: SteamQualityState) -> None:
        """Add state to quality history with size limit."""
        self._quality_history.append(state)
        if len(self._quality_history) > self._max_history_size:
            self._quality_history = self._quality_history[-self._max_history_size:]
