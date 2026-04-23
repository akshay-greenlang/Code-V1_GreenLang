# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Operational Guardrails

Implements safety guardrails to ensure the system operates within
safe boundaries and provides recommendations rather than autonomous actions.

Operational Guardrails:
1. Never bypass Safety Instrumented Systems (SIS)
2. No direct control-loop manipulation
3. Recommendations only, not autonomous actions
4. Out-of-distribution input detection
5. Uncertainty gating (high uncertainty -> conservative recommendations)

Safety Principles:
- Never present predictions as certainties
- Fail safe on poor data quality
- Request engineering review when outside training distribution
- No sensitive OT data export without authorization

Standards Reference:
- IEC 61511: Safety Instrumented Systems for Process Industries
- IEC 61508: Functional Safety of E/E/PE Systems
- ISA-84: Application of Safety Instrumented Systems

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator

from .exceptions import (
    ExchangerproSafetyError,
    SISBypassAttemptError,
    ControlLoopManipulationError,
    UnauthorizedDataExportError,
    ModelPredictionError,
    ViolationContext,
    ViolationDetails,
    ViolationSeverity,
    SafetyDomain,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class ActionType(str, Enum):
    """Types of actions the system can take."""

    RECOMMENDATION = "recommendation"  # Advisory only
    ALERT = "alert"  # Notify operators
    INFORMATION = "information"  # Display information
    DATA_EXPORT = "data_export"  # Export data
    CONTROL_SETPOINT = "control_setpoint"  # BLOCKED: Direct control
    SIS_OVERRIDE = "sis_override"  # BLOCKED: SIS bypass


class GuardrailDecision(str, Enum):
    """Decision outcome from guardrail check."""

    ALLOW = "allow"  # Action is permitted
    ALLOW_WITH_WARNING = "allow_with_warning"  # Permitted but flagged
    REQUIRE_REVIEW = "require_review"  # Needs human review
    BLOCK = "block"  # Action is blocked
    ESCALATE = "escalate"  # Escalate to higher authority


class UncertaintyLevel(str, Enum):
    """Classification of prediction uncertainty."""

    LOW = "low"  # < 0.15 - High confidence
    MEDIUM = "medium"  # 0.15 - 0.35 - Moderate confidence
    HIGH = "high"  # 0.35 - 0.60 - Low confidence
    CRITICAL = "critical"  # > 0.60 - Very low confidence


class DistributionStatus(str, Enum):
    """Status of input relative to training distribution."""

    IN_DISTRIBUTION = "in_distribution"  # Within training data range
    EDGE_CASE = "edge_case"  # Near boundary of training data
    OUT_OF_DISTRIBUTION = "out_of_distribution"  # Outside training data
    ANOMALOUS = "anomalous"  # Significantly different from training data


class ConservativeMode(str, Enum):
    """Conservative recommendation modes."""

    NORMAL = "normal"  # Standard recommendations
    CAUTIOUS = "cautious"  # More conservative margins
    HIGHLY_CONSERVATIVE = "highly_conservative"  # Maximum safety margins
    DEFER_TO_HUMAN = "defer_to_human"  # No automated recommendations


# =============================================================================
# CONFIGURATION
# =============================================================================


class UncertaintyThresholds(BaseModel):
    """
    Thresholds for uncertainty-based gating.

    These thresholds determine how predictions are handled based
    on their uncertainty levels.
    """

    # Uncertainty level thresholds
    low_max: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Maximum uncertainty for LOW level"
    )
    medium_max: float = Field(
        default=0.35,
        ge=0.15,
        le=0.7,
        description="Maximum uncertainty for MEDIUM level"
    )
    high_max: float = Field(
        default=0.60,
        ge=0.35,
        le=0.9,
        description="Maximum uncertainty for HIGH level"
    )

    # Gating thresholds
    auto_approve_max: float = Field(
        default=0.10,
        ge=0.0,
        le=0.3,
        description="Max uncertainty for auto-approval"
    )
    human_review_min: float = Field(
        default=0.25,
        ge=0.1,
        le=0.5,
        description="Min uncertainty requiring human review"
    )
    block_min: float = Field(
        default=0.70,
        ge=0.5,
        le=1.0,
        description="Min uncertainty to block action"
    )

    # Confidence interval thresholds
    max_ci_width_for_auto: float = Field(
        default=0.15,
        ge=0.05,
        le=0.5,
        description="Max CI width for auto-approval"
    )


class DistributionBounds(BaseModel):
    """
    Bounds for detecting out-of-distribution inputs.

    Based on training data statistics for each feature.
    """

    # Temperature bounds (C)
    temperature_min_C: float = Field(default=-20.0)
    temperature_max_C: float = Field(default=500.0)

    # Flow rate bounds (kg/s)
    flow_rate_min_kg_s: float = Field(default=0.0)
    flow_rate_max_kg_s: float = Field(default=1000.0)

    # Pressure bounds (kPa)
    pressure_min_kPa: float = Field(default=0.0)
    pressure_max_kPa: float = Field(default=10000.0)

    # Effectiveness bounds
    effectiveness_min: float = Field(default=0.0)
    effectiveness_max: float = Field(default=1.0)

    # Fouling resistance bounds (m2K/W)
    fouling_min: float = Field(default=0.0)
    fouling_max: float = Field(default=0.01)

    # Standard deviation multiplier for edge case detection
    edge_case_std_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Number of std devs for edge case"
    )

    # Standard deviation multiplier for OOD detection
    ood_std_multiplier: float = Field(
        default=3.0,
        ge=2.0,
        le=10.0,
        description="Number of std devs for OOD"
    )


class GuardrailsConfig(BaseModel):
    """
    Configuration for operational guardrails.

    Attributes:
        uncertainty_thresholds: Thresholds for uncertainty gating
        distribution_bounds: Bounds for OOD detection
        blocked_action_types: Action types that are always blocked
        require_authorization_for: Action types requiring authorization
        conservative_mode_threshold: Uncertainty above which conservative mode activates
        max_autonomous_urgency: Maximum urgency score for autonomous recommendations
        data_export_requires_auth: Whether data export needs authorization
        log_all_decisions: Log all guardrail decisions
    """

    uncertainty_thresholds: UncertaintyThresholds = Field(
        default_factory=UncertaintyThresholds
    )
    distribution_bounds: DistributionBounds = Field(
        default_factory=DistributionBounds
    )

    # Action type restrictions
    blocked_action_types: List[ActionType] = Field(
        default_factory=lambda: [
            ActionType.CONTROL_SETPOINT,
            ActionType.SIS_OVERRIDE,
        ],
        description="Action types that are always blocked"
    )
    require_authorization_for: List[ActionType] = Field(
        default_factory=lambda: [ActionType.DATA_EXPORT],
        description="Action types requiring explicit authorization"
    )

    # Conservative mode settings
    conservative_mode_threshold: float = Field(
        default=0.40,
        ge=0.2,
        le=0.8,
        description="Uncertainty threshold for conservative mode"
    )
    max_autonomous_urgency: float = Field(
        default=0.80,
        ge=0.5,
        le=1.0,
        description="Maximum urgency for autonomous recommendations"
    )

    # Authorization settings
    data_export_requires_auth: bool = Field(
        default=True,
        description="Require authorization for OT data export"
    )

    # Logging
    log_all_decisions: bool = Field(
        default=True,
        description="Log all guardrail decisions for audit"
    )

    # SIS protection list
    protected_sis_functions: List[str] = Field(
        default_factory=lambda: [
            "high_temperature_trip",
            "low_flow_trip",
            "high_pressure_trip",
            "emergency_shutdown",
            "safety_interlock",
        ],
        description="SIS functions that cannot be bypassed"
    )

    # Protected control loops
    protected_control_loops: List[str] = Field(
        default_factory=lambda: [
            "temperature_controller",
            "flow_controller",
            "pressure_controller",
            "level_controller",
        ],
        description="Control loops that cannot be directly manipulated"
    )


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class GuardrailCheckResult:
    """
    Result of a guardrail check.

    Attributes:
        action_type: Type of action being checked
        decision: Guardrail decision
        reason: Reason for the decision
        uncertainty_level: Uncertainty classification
        distribution_status: OOD status
        conservative_mode: Active conservative mode
        requires_authorization: Whether authorization is needed
        authorizing_role: Role required for authorization
        blocked_by: Which guardrail blocked (if blocked)
        recommendations: Modified recommendations (if any)
        timestamp: When check was performed
        provenance_hash: SHA-256 hash for audit
    """

    action_type: ActionType
    decision: GuardrailDecision
    reason: str
    uncertainty_level: UncertaintyLevel = UncertaintyLevel.LOW
    distribution_status: DistributionStatus = DistributionStatus.IN_DISTRIBUTION
    conservative_mode: ConservativeMode = ConservativeMode.NORMAL
    requires_authorization: bool = False
    authorizing_role: Optional[str] = None
    blocked_by: Optional[str] = None
    recommendations: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            content = (
                f"{self.action_type.value}|{self.decision.value}|"
                f"{self.uncertainty_level.value}|{self.distribution_status.value}|"
                f"{self.timestamp.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class InputFeatures:
    """
    Input features for OOD detection.

    Attributes:
        exchanger_id: Heat exchanger identifier
        temperatures: Dict of temperature readings
        flow_rates: Dict of flow rate readings
        pressures: Dict of pressure readings
        effectiveness: Heat exchanger effectiveness
        fouling_resistance: Fouling resistance
        additional_features: Any additional features
    """

    exchanger_id: str
    temperatures: Dict[str, float] = field(default_factory=dict)
    flow_rates: Dict[str, float] = field(default_factory=dict)
    pressures: Dict[str, float] = field(default_factory=dict)
    effectiveness: Optional[float] = None
    fouling_resistance: Optional[float] = None
    additional_features: Dict[str, float] = field(default_factory=dict)


@dataclass
class UncertaintyEstimate:
    """
    Uncertainty estimate for a prediction.

    Attributes:
        epistemic: Model uncertainty (reducible with more data)
        aleatoric: Data uncertainty (irreducible noise)
        total: Combined uncertainty
        confidence_interval: (lower, upper) bounds
        calibration_score: How well-calibrated the uncertainty is
    """

    epistemic: float = 0.0
    aleatoric: float = 0.0
    total: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    calibration_score: float = 1.0

    @property
    def ci_width(self) -> float:
        """Width of confidence interval."""
        return self.confidence_interval[1] - self.confidence_interval[0]


@dataclass
class RecommendationAdjustment:
    """
    Adjustment to a recommendation based on guardrails.

    Attributes:
        original_urgency: Original urgency score
        adjusted_urgency: Adjusted urgency score
        original_action: Original recommended action
        adjusted_action: Adjusted recommended action
        adjustment_reason: Why adjustment was made
        conservative_mode: Active conservative mode
        requires_review: Whether human review is required
        review_deadline: Deadline for human review
    """

    original_urgency: float
    adjusted_urgency: float
    original_action: str
    adjusted_action: str
    adjustment_reason: str
    conservative_mode: ConservativeMode
    requires_review: bool = False
    review_deadline: Optional[datetime] = None


# =============================================================================
# OPERATIONAL GUARDRAILS
# =============================================================================


class OperationalGuardrails:
    """
    Implements operational safety guardrails for the Exchangerpro system.

    This class enforces critical safety boundaries:
    1. Blocks any attempt to bypass SIS or manipulate control loops
    2. Detects out-of-distribution inputs and flags for review
    3. Applies uncertainty gating for conservative recommendations
    4. Ensures all outputs are recommendations, not autonomous actions
    5. Controls data export authorization

    Safety Principles:
    - Never present predictions as certainties
    - Fail safe on poor data quality
    - Request engineering review when outside training distribution
    - No sensitive OT data export without authorization

    Example:
        >>> config = GuardrailsConfig()
        >>> guardrails = OperationalGuardrails(config)
        >>>
        >>> # Check if an action is allowed
        >>> result = guardrails.check_action(
        ...     action_type=ActionType.RECOMMENDATION,
        ...     uncertainty=UncertaintyEstimate(total=0.25),
        ...     input_features=features,
        ... )
        >>>
        >>> if result.decision == GuardrailDecision.BLOCK:
        ...     raise SafetyError(result.reason)

    Author: GL-BackendDeveloper
    Version: 1.0.0
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        config: Optional[GuardrailsConfig] = None,
        decision_callback: Optional[Callable[[GuardrailCheckResult], None]] = None,
    ) -> None:
        """
        Initialize operational guardrails.

        Args:
            config: Guardrails configuration
            decision_callback: Optional callback for all decisions
        """
        self.config = config or GuardrailsConfig()
        self._lock = threading.RLock()
        self._decision_history: List[GuardrailCheckResult] = []
        self._decision_callbacks: List[Callable[[GuardrailCheckResult], None]] = []
        self._authorized_sessions: Set[str] = set()

        if decision_callback:
            self._decision_callbacks.append(decision_callback)

        logger.info(
            f"OperationalGuardrails initialized: "
            f"blocked_actions={[a.value for a in self.config.blocked_action_types]}, "
            f"conservative_threshold={self.config.conservative_mode_threshold}"
        )

    # =========================================================================
    # ACTION TYPE CHECKS
    # =========================================================================

    def check_action(
        self,
        action_type: ActionType,
        uncertainty: Optional[UncertaintyEstimate] = None,
        input_features: Optional[InputFeatures] = None,
        urgency_score: Optional[float] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> GuardrailCheckResult:
        """
        Check if an action is allowed by guardrails.

        Args:
            action_type: Type of action to check
            uncertainty: Uncertainty estimate for prediction
            input_features: Input features for OOD detection
            urgency_score: Urgency score of recommendation
            session_id: Session ID for authorization tracking
            context: Additional context

        Returns:
            GuardrailCheckResult with decision and details
        """
        with self._lock:
            # Check blocked action types first
            if action_type in self.config.blocked_action_types:
                result = self._create_blocked_result(action_type)
                self._record_decision(result)
                return result

            # Check authorization requirements
            if action_type in self.config.require_authorization_for:
                if not self._check_authorization(action_type, session_id):
                    result = self._create_auth_required_result(action_type)
                    self._record_decision(result)
                    return result

            # Determine uncertainty level
            uncertainty = uncertainty or UncertaintyEstimate()
            uncertainty_level = self._classify_uncertainty(uncertainty)

            # Check distribution status
            distribution_status = DistributionStatus.IN_DISTRIBUTION
            if input_features:
                distribution_status = self._check_distribution(input_features)

            # Determine conservative mode
            conservative_mode = self._determine_conservative_mode(
                uncertainty, distribution_status
            )

            # Apply uncertainty gating
            decision, reason = self._apply_uncertainty_gating(
                action_type, uncertainty, uncertainty_level, distribution_status
            )

            # Check urgency constraints
            if urgency_score is not None and urgency_score > self.config.max_autonomous_urgency:
                if decision == GuardrailDecision.ALLOW:
                    decision = GuardrailDecision.REQUIRE_REVIEW
                    reason = (
                        f"Urgency score {urgency_score:.2f} exceeds autonomous limit "
                        f"{self.config.max_autonomous_urgency:.2f}"
                    )

            # Build result
            result = GuardrailCheckResult(
                action_type=action_type,
                decision=decision,
                reason=reason,
                uncertainty_level=uncertainty_level,
                distribution_status=distribution_status,
                conservative_mode=conservative_mode,
                requires_authorization=action_type in self.config.require_authorization_for,
            )

            self._record_decision(result)
            return result

    def _create_blocked_result(self, action_type: ActionType) -> GuardrailCheckResult:
        """Create result for blocked action types."""
        if action_type == ActionType.SIS_OVERRIDE:
            reason = (
                "SIS override is NEVER allowed. Safety Instrumented Systems "
                "must not be bypassed under any circumstances."
            )
            blocked_by = "SIS_PROTECTION"
        elif action_type == ActionType.CONTROL_SETPOINT:
            reason = (
                "Direct control setpoint manipulation is not allowed. "
                "The system provides recommendations only."
            )
            blocked_by = "RECOMMENDATION_ONLY"
        else:
            reason = f"Action type {action_type.value} is blocked by policy."
            blocked_by = "POLICY"

        return GuardrailCheckResult(
            action_type=action_type,
            decision=GuardrailDecision.BLOCK,
            reason=reason,
            blocked_by=blocked_by,
        )

    def _create_auth_required_result(
        self,
        action_type: ActionType,
    ) -> GuardrailCheckResult:
        """Create result for actions requiring authorization."""
        return GuardrailCheckResult(
            action_type=action_type,
            decision=GuardrailDecision.REQUIRE_REVIEW,
            reason=f"Action type {action_type.value} requires explicit authorization.",
            requires_authorization=True,
            authorizing_role="data_steward" if action_type == ActionType.DATA_EXPORT else "supervisor",
        )

    # =========================================================================
    # SIS AND CONTROL LOOP PROTECTION
    # =========================================================================

    def check_sis_interaction(
        self,
        sis_function: str,
        intended_action: str,
        exchanger_id: str,
    ) -> GuardrailCheckResult:
        """
        Check if an interaction with SIS is allowed.

        SIS bypass is NEVER allowed. This is a hard safety constraint.

        Args:
            sis_function: The SIS function being interacted with
            intended_action: What action is intended
            exchanger_id: Related exchanger

        Returns:
            GuardrailCheckResult (always BLOCK for bypass attempts)

        Raises:
            SISBypassAttemptError: If bypass is attempted
        """
        with self._lock:
            # Check if this is a protected SIS function
            if sis_function.lower() in [s.lower() for s in self.config.protected_sis_functions]:
                # Any non-read action is blocked
                if intended_action.lower() not in ["read", "monitor", "status"]:
                    context = ViolationContext(exchanger_id=exchanger_id)

                    result = GuardrailCheckResult(
                        action_type=ActionType.SIS_OVERRIDE,
                        decision=GuardrailDecision.BLOCK,
                        reason=(
                            f"BLOCKED: Attempted {intended_action} on SIS function "
                            f"'{sis_function}'. SIS bypass is NEVER allowed."
                        ),
                        blocked_by="SIS_PROTECTION",
                    )
                    self._record_decision(result)

                    # Also raise exception for immediate handling
                    raise SISBypassAttemptError(
                        sis_function=sis_function,
                        attempted_action=intended_action,
                        context=context,
                    )

            # Read-only access is allowed
            return GuardrailCheckResult(
                action_type=ActionType.INFORMATION,
                decision=GuardrailDecision.ALLOW,
                reason=f"Read-only access to SIS function '{sis_function}' is permitted.",
            )

    def check_control_loop_interaction(
        self,
        control_loop: str,
        intended_action: str,
        setpoint_value: Optional[float] = None,
        exchanger_id: str = "unknown",
    ) -> GuardrailCheckResult:
        """
        Check if an interaction with a control loop is allowed.

        Direct setpoint manipulation is NEVER allowed.
        Recommendations only.

        Args:
            control_loop: The control loop being interacted with
            intended_action: What action is intended
            setpoint_value: Proposed setpoint (if any)
            exchanger_id: Related exchanger

        Returns:
            GuardrailCheckResult

        Raises:
            ControlLoopManipulationError: If direct control is attempted
        """
        with self._lock:
            # Check if this is a protected control loop
            is_protected = any(
                cl.lower() in control_loop.lower()
                for cl in self.config.protected_control_loops
            )

            # Setpoint writes are always blocked
            if intended_action.lower() in ["write", "set", "change", "modify"]:
                context = ViolationContext(exchanger_id=exchanger_id)

                result = GuardrailCheckResult(
                    action_type=ActionType.CONTROL_SETPOINT,
                    decision=GuardrailDecision.BLOCK,
                    reason=(
                        f"BLOCKED: Direct manipulation of control loop '{control_loop}' "
                        f"is not allowed. System provides recommendations only."
                    ),
                    blocked_by="RECOMMENDATION_ONLY",
                )
                self._record_decision(result)

                raise ControlLoopManipulationError(
                    control_loop=control_loop,
                    attempted_setpoint=setpoint_value,
                    context=context,
                )

            # Recommendations about control loops are allowed
            if intended_action.lower() in ["recommend", "suggest", "advise"]:
                return GuardrailCheckResult(
                    action_type=ActionType.RECOMMENDATION,
                    decision=GuardrailDecision.ALLOW,
                    reason=(
                        f"Recommendation about control loop '{control_loop}' is permitted. "
                        f"Operator must implement changes manually."
                    ),
                )

            # Read access is allowed
            return GuardrailCheckResult(
                action_type=ActionType.INFORMATION,
                decision=GuardrailDecision.ALLOW,
                reason=f"Read access to control loop '{control_loop}' is permitted.",
            )

    # =========================================================================
    # UNCERTAINTY GATING
    # =========================================================================

    def _classify_uncertainty(
        self,
        uncertainty: UncertaintyEstimate,
    ) -> UncertaintyLevel:
        """Classify total uncertainty into levels."""
        thresholds = self.config.uncertainty_thresholds
        total = uncertainty.total

        if total <= thresholds.low_max:
            return UncertaintyLevel.LOW
        elif total <= thresholds.medium_max:
            return UncertaintyLevel.MEDIUM
        elif total <= thresholds.high_max:
            return UncertaintyLevel.HIGH
        else:
            return UncertaintyLevel.CRITICAL

    def _apply_uncertainty_gating(
        self,
        action_type: ActionType,
        uncertainty: UncertaintyEstimate,
        uncertainty_level: UncertaintyLevel,
        distribution_status: DistributionStatus,
    ) -> Tuple[GuardrailDecision, str]:
        """
        Apply uncertainty-based gating to determine decision.

        Returns:
            Tuple of (decision, reason)
        """
        thresholds = self.config.uncertainty_thresholds
        total = uncertainty.total
        ci_width = uncertainty.ci_width

        # Check for blocking conditions
        if total >= thresholds.block_min:
            return (
                GuardrailDecision.BLOCK,
                f"Uncertainty {total:.2f} exceeds blocking threshold "
                f"{thresholds.block_min:.2f}. Cannot make reliable recommendation."
            )

        # Check for OOD - always require review
        if distribution_status == DistributionStatus.OUT_OF_DISTRIBUTION:
            return (
                GuardrailDecision.REQUIRE_REVIEW,
                "Input is outside training distribution. "
                "Engineering review required before acting on recommendation."
            )

        if distribution_status == DistributionStatus.ANOMALOUS:
            return (
                GuardrailDecision.ESCALATE,
                "Input appears anomalous. Escalate to process engineer for assessment."
            )

        # Check for human review threshold
        if total >= thresholds.human_review_min:
            return (
                GuardrailDecision.REQUIRE_REVIEW,
                f"Uncertainty {total:.2f} requires human review "
                f"(threshold: {thresholds.human_review_min:.2f})."
            )

        # Check CI width for auto-approval
        if ci_width > thresholds.max_ci_width_for_auto:
            return (
                GuardrailDecision.ALLOW_WITH_WARNING,
                f"Confidence interval width {ci_width:.2f} is wide. "
                f"Recommendation provided with reduced confidence."
            )

        # Check for auto-approval
        if total <= thresholds.auto_approve_max and ci_width <= thresholds.max_ci_width_for_auto:
            return (
                GuardrailDecision.ALLOW,
                f"Low uncertainty {total:.2f} with narrow CI. "
                f"High-confidence recommendation."
            )

        # Default: allow with warning for edge cases
        if distribution_status == DistributionStatus.EDGE_CASE:
            return (
                GuardrailDecision.ALLOW_WITH_WARNING,
                "Input is near edge of training distribution. "
                "Recommendation may have reduced accuracy."
            )

        return (
            GuardrailDecision.ALLOW,
            f"Uncertainty {total:.2f} is within acceptable range."
        )

    def _determine_conservative_mode(
        self,
        uncertainty: UncertaintyEstimate,
        distribution_status: DistributionStatus,
    ) -> ConservativeMode:
        """Determine appropriate conservative mode."""
        # OOD always triggers highly conservative
        if distribution_status in (
            DistributionStatus.OUT_OF_DISTRIBUTION,
            DistributionStatus.ANOMALOUS,
        ):
            return ConservativeMode.DEFER_TO_HUMAN

        # High uncertainty triggers conservative mode
        if uncertainty.total >= self.config.conservative_mode_threshold:
            if uncertainty.total >= 0.6:
                return ConservativeMode.HIGHLY_CONSERVATIVE
            else:
                return ConservativeMode.CAUTIOUS

        # Edge cases warrant caution
        if distribution_status == DistributionStatus.EDGE_CASE:
            return ConservativeMode.CAUTIOUS

        return ConservativeMode.NORMAL

    # =========================================================================
    # OUT-OF-DISTRIBUTION DETECTION
    # =========================================================================

    def _check_distribution(self, features: InputFeatures) -> DistributionStatus:
        """
        Check if input features are within training distribution.

        Uses simple bounds checking. In production, this would use
        more sophisticated methods like:
        - Mahalanobis distance
        - Isolation Forest
        - Reconstruction error from autoencoder
        """
        bounds = self.config.distribution_bounds
        ood_count = 0
        edge_count = 0
        total_checks = 0

        # Check temperatures
        for name, temp in features.temperatures.items():
            total_checks += 1
            if temp < bounds.temperature_min_C or temp > bounds.temperature_max_C:
                ood_count += 1
            elif temp < bounds.temperature_min_C * 1.1 or temp > bounds.temperature_max_C * 0.9:
                edge_count += 1

        # Check flow rates
        for name, flow in features.flow_rates.items():
            total_checks += 1
            if flow < bounds.flow_rate_min_kg_s or flow > bounds.flow_rate_max_kg_s:
                ood_count += 1
            elif flow > bounds.flow_rate_max_kg_s * 0.9:
                edge_count += 1

        # Check pressures
        for name, pressure in features.pressures.items():
            total_checks += 1
            if pressure < bounds.pressure_min_kPa or pressure > bounds.pressure_max_kPa:
                ood_count += 1

        # Check effectiveness
        if features.effectiveness is not None:
            total_checks += 1
            if features.effectiveness < bounds.effectiveness_min or features.effectiveness > bounds.effectiveness_max:
                ood_count += 1
            elif features.effectiveness > 0.95:  # Very high effectiveness is suspicious
                edge_count += 1

        # Check fouling
        if features.fouling_resistance is not None:
            total_checks += 1
            if features.fouling_resistance > bounds.fouling_max:
                ood_count += 1
            elif features.fouling_resistance > bounds.fouling_max * 0.8:
                edge_count += 1

        # Determine status
        if total_checks == 0:
            return DistributionStatus.IN_DISTRIBUTION

        ood_ratio = ood_count / total_checks
        edge_ratio = edge_count / total_checks

        if ood_ratio > 0.3:
            return DistributionStatus.ANOMALOUS
        elif ood_count > 0:
            return DistributionStatus.OUT_OF_DISTRIBUTION
        elif edge_ratio > 0.3:
            return DistributionStatus.EDGE_CASE
        else:
            return DistributionStatus.IN_DISTRIBUTION

    def detect_ood(
        self,
        features: InputFeatures,
    ) -> Tuple[DistributionStatus, Dict[str, Any]]:
        """
        Detect out-of-distribution inputs with detailed report.

        Args:
            features: Input features to check

        Returns:
            Tuple of (status, details_dict)
        """
        with self._lock:
            status = self._check_distribution(features)

            details = {
                "exchanger_id": features.exchanger_id,
                "status": status.value,
                "features_checked": {
                    "temperatures": list(features.temperatures.keys()),
                    "flow_rates": list(features.flow_rates.keys()),
                    "pressures": list(features.pressures.keys()),
                    "effectiveness": features.effectiveness is not None,
                    "fouling_resistance": features.fouling_resistance is not None,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if status in (DistributionStatus.OUT_OF_DISTRIBUTION, DistributionStatus.ANOMALOUS):
                logger.warning(
                    f"OOD input detected for {features.exchanger_id}: status={status.value}"
                )
                details["recommendation"] = (
                    "Request engineering review before acting on predictions."
                )

            return status, details

    # =========================================================================
    # RECOMMENDATION ADJUSTMENT
    # =========================================================================

    def adjust_recommendation(
        self,
        urgency_score: float,
        recommended_action: str,
        uncertainty: UncertaintyEstimate,
        distribution_status: DistributionStatus,
    ) -> RecommendationAdjustment:
        """
        Adjust a recommendation based on guardrails.

        High uncertainty or OOD status results in more conservative
        recommendations.

        Args:
            urgency_score: Original urgency score
            recommended_action: Original recommended action
            uncertainty: Uncertainty estimate
            distribution_status: OOD status

        Returns:
            RecommendationAdjustment with adjusted values
        """
        conservative_mode = self._determine_conservative_mode(
            uncertainty, distribution_status
        )

        adjusted_urgency = urgency_score
        adjusted_action = recommended_action
        adjustment_reason = "No adjustment needed"
        requires_review = False
        review_deadline = None

        if conservative_mode == ConservativeMode.DEFER_TO_HUMAN:
            adjusted_urgency = min(urgency_score, 0.3)  # Cap urgency
            adjusted_action = f"[REVIEW REQUIRED] {recommended_action}"
            adjustment_reason = "Outside training distribution - human review required"
            requires_review = True
            review_deadline = datetime.now(timezone.utc)

        elif conservative_mode == ConservativeMode.HIGHLY_CONSERVATIVE:
            # Reduce urgency by 30%
            adjusted_urgency = urgency_score * 0.7
            adjusted_action = f"[HIGH UNCERTAINTY] {recommended_action}"
            adjustment_reason = "High uncertainty - applying conservative margin"
            requires_review = urgency_score > 0.6

        elif conservative_mode == ConservativeMode.CAUTIOUS:
            # Reduce urgency by 15%
            adjusted_urgency = urgency_score * 0.85
            adjustment_reason = "Elevated uncertainty - applying caution margin"
            requires_review = urgency_score > 0.8

        # Never exceed max autonomous urgency
        if adjusted_urgency > self.config.max_autonomous_urgency:
            adjusted_urgency = self.config.max_autonomous_urgency
            requires_review = True
            adjustment_reason += f"; capped at {self.config.max_autonomous_urgency}"

        return RecommendationAdjustment(
            original_urgency=urgency_score,
            adjusted_urgency=adjusted_urgency,
            original_action=recommended_action,
            adjusted_action=adjusted_action,
            adjustment_reason=adjustment_reason,
            conservative_mode=conservative_mode,
            requires_review=requires_review,
            review_deadline=review_deadline,
        )

    # =========================================================================
    # DATA EXPORT AUTHORIZATION
    # =========================================================================

    def check_data_export(
        self,
        data_type: str,
        destination: str,
        session_id: Optional[str] = None,
        exchanger_id: str = "unknown",
    ) -> GuardrailCheckResult:
        """
        Check if data export is authorized.

        Args:
            data_type: Type of data being exported
            destination: Export destination
            session_id: Session ID for authorization check
            exchanger_id: Related exchanger

        Returns:
            GuardrailCheckResult

        Raises:
            UnauthorizedDataExportError: If export is not authorized
        """
        with self._lock:
            if not self.config.data_export_requires_auth:
                return GuardrailCheckResult(
                    action_type=ActionType.DATA_EXPORT,
                    decision=GuardrailDecision.ALLOW,
                    reason="Data export authorization not required by configuration.",
                )

            # Check if session is authorized
            if session_id and session_id in self._authorized_sessions:
                return GuardrailCheckResult(
                    action_type=ActionType.DATA_EXPORT,
                    decision=GuardrailDecision.ALLOW,
                    reason=f"Data export authorized for session {session_id}.",
                )

            # Sensitive OT data requires authorization
            context = ViolationContext(exchanger_id=exchanger_id)

            result = GuardrailCheckResult(
                action_type=ActionType.DATA_EXPORT,
                decision=GuardrailDecision.REQUIRE_REVIEW,
                reason=(
                    f"Export of '{data_type}' to '{destination}' requires authorization. "
                    f"OT data export is restricted."
                ),
                requires_authorization=True,
                authorizing_role="data_steward",
            )
            self._record_decision(result)

            # Raise exception for immediate handling
            raise UnauthorizedDataExportError(
                data_type=data_type,
                destination=destination,
                context=context,
            )

    def authorize_session(self, session_id: str, authorized_by: str) -> None:
        """
        Authorize a session for data export.

        Args:
            session_id: Session to authorize
            authorized_by: Who is authorizing
        """
        with self._lock:
            self._authorized_sessions.add(session_id)
            logger.info(
                f"Session {session_id} authorized for data export by {authorized_by}"
            )

    def revoke_session(self, session_id: str) -> bool:
        """
        Revoke authorization for a session.

        Args:
            session_id: Session to revoke

        Returns:
            True if session was authorized and is now revoked
        """
        with self._lock:
            if session_id in self._authorized_sessions:
                self._authorized_sessions.remove(session_id)
                logger.info(f"Session {session_id} authorization revoked")
                return True
            return False

    # =========================================================================
    # AUTHORIZATION CHECKING
    # =========================================================================

    def _check_authorization(
        self,
        action_type: ActionType,
        session_id: Optional[str],
    ) -> bool:
        """Check if action is authorized for session."""
        if action_type == ActionType.DATA_EXPORT:
            return session_id is not None and session_id in self._authorized_sessions
        return True

    # =========================================================================
    # DECISION RECORDING
    # =========================================================================

    def _record_decision(self, result: GuardrailCheckResult) -> None:
        """Record decision for audit trail."""
        self._decision_history.append(result)

        # Trim history if needed
        max_history = 10000
        if len(self._decision_history) > max_history:
            self._decision_history = self._decision_history[-max_history:]

        # Log if configured
        if self.config.log_all_decisions:
            log_level = logging.INFO
            if result.decision == GuardrailDecision.BLOCK:
                log_level = logging.WARNING
            elif result.decision in (GuardrailDecision.REQUIRE_REVIEW, GuardrailDecision.ESCALATE):
                log_level = logging.INFO

            logger.log(
                log_level,
                f"Guardrail decision: action={result.action_type.value}, "
                f"decision={result.decision.value}, reason={result.reason[:100]}"
            )

        # Invoke callbacks
        for callback in self._decision_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Decision callback failed: {e}")

    def register_decision_callback(
        self,
        callback: Callable[[GuardrailCheckResult], None],
    ) -> None:
        """Register callback for guardrail decisions."""
        if callable(callback):
            self._decision_callbacks.append(callback)

    def get_decision_history(
        self,
        action_type: Optional[ActionType] = None,
        decision: Optional[GuardrailDecision] = None,
        limit: int = 100,
    ) -> List[GuardrailCheckResult]:
        """
        Get decision history with optional filtering.

        Args:
            action_type: Filter by action type
            decision: Filter by decision
            limit: Maximum entries to return

        Returns:
            List of GuardrailCheckResult
        """
        with self._lock:
            results = self._decision_history.copy()

            if action_type:
                results = [r for r in results if r.action_type == action_type]

            if decision:
                results = [r for r in results if r.decision == decision]

            return list(reversed(results[-limit:]))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def check_is_recommendation_only(action_type: ActionType) -> bool:
    """
    Quick check if action type is recommendation-only (allowed).

    Args:
        action_type: Action type to check

    Returns:
        True if action is allowed, False if blocked
    """
    blocked = {ActionType.CONTROL_SETPOINT, ActionType.SIS_OVERRIDE}
    return action_type not in blocked


def get_conservative_urgency(
    urgency: float,
    uncertainty: float,
    is_ood: bool = False,
) -> float:
    """
    Get conservative urgency based on uncertainty.

    Args:
        urgency: Original urgency score
        uncertainty: Uncertainty estimate
        is_ood: Whether input is out-of-distribution

    Returns:
        Adjusted urgency score
    """
    if is_ood:
        return min(urgency * 0.5, 0.3)

    if uncertainty > 0.6:
        return urgency * 0.7
    elif uncertainty > 0.4:
        return urgency * 0.85
    else:
        return urgency


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ActionType",
    "GuardrailDecision",
    "UncertaintyLevel",
    "DistributionStatus",
    "ConservativeMode",
    # Config
    "UncertaintyThresholds",
    "DistributionBounds",
    "GuardrailsConfig",
    # Data models
    "GuardrailCheckResult",
    "InputFeatures",
    "UncertaintyEstimate",
    "RecommendationAdjustment",
    # Main class
    "OperationalGuardrails",
    # Convenience functions
    "check_is_recommendation_only",
    "get_conservative_urgency",
]
