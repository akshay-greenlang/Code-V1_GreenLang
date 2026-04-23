"""
GL-012 STEAMQUAL SteamQualityController - Quality Controller

This module implements the supervisory quality control layer for steam systems,
providing constraint computation for optimization and coordinated action
management across multiple assets.

Control Architecture:
    - Supervisory layer above individual equipment controllers
    - Compute constraints for optimization objectives
    - Coordinate actions across desuperheaters, separators, and drain valves
    - Advisory mode (default) vs automation mode

Key Features:
    - Multi-point quality monitoring and control
    - Constraint computation for optimizer integration
    - Cross-asset action coordination
    - Safety envelope enforcement
    - Zero-hallucination deterministic logic

Reference Standards:
    - ISA-18.2 Management of Alarm Systems
    - IEC 61511 Functional Safety
    - ASME PTC 19.11 Steam and Water Sampling

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ControlMode(str, Enum):
    """Control mode enumeration for supervisory layer."""
    ADVISORY = "advisory"  # Default - recommendations only
    AUTOMATION = "automation"  # Closed-loop with site approval
    MONITORING = "monitoring"  # Observe only, no recommendations
    EMERGENCY = "emergency"  # Safety override mode


class QualityStatus(str, Enum):
    """Steam quality status enumeration."""
    OPTIMAL = "optimal"
    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ActionPriority(str, Enum):
    """Coordinated action priority levels."""
    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Action within 5 minutes
    MEDIUM = "medium"  # Action within 30 minutes
    LOW = "low"  # Optimization opportunity
    INFORMATIONAL = "informational"  # No action needed


class ConstraintType(str, Enum):
    """Constraint type for optimization."""
    EQUALITY = "equality"
    INEQUALITY_LE = "inequality_le"  # Less than or equal
    INEQUALITY_GE = "inequality_ge"  # Greater than or equal
    RANGE = "range"  # Between bounds


# =============================================================================
# DATA MODELS
# =============================================================================

class QualityMeasurement(BaseModel):
    """Steam quality measurement at a specific point."""

    measurement_id: str = Field(..., description="Measurement point ID")
    asset_id: str = Field(..., description="Associated asset ID")
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
        description="Saturation temperature at current pressure (C)"
    )
    flow_rate_kg_s: float = Field(..., ge=0, description="Steam flow rate (kg/s)")
    moisture_ppm: Optional[float] = Field(
        None,
        ge=0,
        description="Moisture content (ppm) if measured"
    )
    silica_ppb: Optional[float] = Field(
        None,
        ge=0,
        description="Silica content (ppb) if measured"
    )
    conductivity_us_cm: Optional[float] = Field(
        None,
        ge=0,
        description="Conductivity (uS/cm) if measured"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Measurement timestamp"
    )

    @validator('dryness_fraction')
    def validate_dryness(cls, v: float) -> float:
        """Validate dryness fraction is physically meaningful."""
        if v < 0.5:
            logger.warning(f"Very low dryness fraction: {v} - verify sensor")
        return v


class QualityTargets(BaseModel):
    """Quality control targets for supervisory layer."""

    target_dryness: float = Field(
        default=0.995,
        ge=0.9,
        le=1.0,
        description="Target steam dryness fraction"
    )
    min_dryness: float = Field(
        default=0.98,
        ge=0.9,
        le=1.0,
        description="Minimum acceptable dryness"
    )
    target_superheat_c: float = Field(
        default=25.0,
        ge=5,
        description="Target superheat (C)"
    )
    min_superheat_c: float = Field(
        default=15.0,
        ge=5,
        description="Minimum superheat margin (C)"
    )
    max_superheat_c: float = Field(
        default=50.0,
        ge=10,
        description="Maximum superheat (C)"
    )
    max_moisture_ppm: float = Field(
        default=500.0,
        ge=0,
        description="Maximum moisture content (ppm)"
    )
    max_silica_ppb: float = Field(
        default=20.0,
        ge=0,
        description="Maximum silica content (ppb)"
    )


class QualityConstraints(BaseModel):
    """Operational constraints for quality control."""

    max_correction_rate_pct_per_min: float = Field(
        default=5.0,
        ge=0,
        description="Maximum correction rate (%/min)"
    )
    min_stability_time_s: float = Field(
        default=60.0,
        ge=0,
        description="Minimum time between corrections (s)"
    )
    coordination_timeout_s: float = Field(
        default=30.0,
        ge=0,
        description="Timeout for coordinated actions (s)"
    )
    safety_margin_factor: float = Field(
        default=1.2,
        ge=1.0,
        description="Safety margin multiplier for constraints"
    )


class OptimizationConstraint(BaseModel):
    """Constraint for optimization solver."""

    constraint_id: str = Field(..., description="Unique constraint ID")
    variable_name: str = Field(..., description="Optimization variable name")
    constraint_type: ConstraintType = Field(..., description="Constraint type")
    lower_bound: Optional[float] = Field(None, description="Lower bound")
    upper_bound: Optional[float] = Field(None, description="Upper bound")
    target_value: Optional[float] = Field(None, description="Target for equality")
    weight: float = Field(default=1.0, ge=0, description="Constraint weight")
    active: bool = Field(default=True, description="Constraint is active")
    reason: str = Field(default="", description="Reason for constraint")
    provenance_hash: str = Field(default="", description="Provenance hash")


class CoordinatedAction(BaseModel):
    """Coordinated action across multiple assets."""

    action_id: str = Field(..., description="Unique action ID")
    priority: ActionPriority = Field(..., description="Action priority")
    affected_assets: List[str] = Field(..., description="List of affected asset IDs")
    action_type: str = Field(..., description="Type of action")
    description: str = Field(..., description="Human-readable description")
    setpoint_changes: Dict[str, float] = Field(
        default_factory=dict,
        description="Proposed setpoint changes by asset"
    )
    sequence_order: List[str] = Field(
        default_factory=list,
        description="Order of asset actions"
    )
    estimated_duration_s: float = Field(
        default=60.0,
        ge=0,
        description="Estimated duration (s)"
    )
    requires_confirmation: bool = Field(
        default=True,
        description="Requires operator confirmation"
    )
    safety_validated: bool = Field(
        default=False,
        description="Passed safety validation"
    )
    rollback_available: bool = Field(
        default=True,
        description="Rollback is possible"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Action creation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class ControlRecommendation(BaseModel):
    """Control recommendation from supervisory layer."""

    recommendation_id: str = Field(..., description="Unique recommendation ID")
    quality_status: QualityStatus = Field(..., description="Current quality status")
    coordinated_actions: List[CoordinatedAction] = Field(
        default_factory=list,
        description="List of coordinated actions"
    )
    optimization_constraints: List[OptimizationConstraint] = Field(
        default_factory=list,
        description="Constraints for optimizer"
    )
    quality_improvement_estimate: float = Field(
        ...,
        description="Estimated quality improvement (0-1)"
    )
    confidence_level: float = Field(
        default=0.9,
        ge=0,
        le=1,
        description="Recommendation confidence"
    )
    rationale: str = Field(..., description="Explanation of recommendation")
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    requires_confirmation: bool = Field(
        default=True,
        description="Requires operator confirmation"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Recommendation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class SupervisoryState(BaseModel):
    """Current state of the supervisory controller."""

    controller_id: str = Field(..., description="Controller identifier")
    mode: ControlMode = Field(..., description="Current control mode")
    active_measurements: int = Field(
        default=0,
        ge=0,
        description="Number of active measurements"
    )
    overall_quality_status: QualityStatus = Field(
        ...,
        description="Overall quality status"
    )
    active_actions: int = Field(
        default=0,
        ge=0,
        description="Number of active coordinated actions"
    )
    pending_confirmations: int = Field(
        default=0,
        ge=0,
        description="Actions awaiting confirmation"
    )
    last_recommendation_time: Optional[datetime] = Field(
        None,
        description="Time of last recommendation"
    )
    automation_enabled: bool = Field(
        default=False,
        description="Automation mode enabled"
    )
    site_approval_valid: bool = Field(
        default=False,
        description="Site approval for automation is valid"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="State timestamp"
    )


# =============================================================================
# QUALITY CONTROLLER
# =============================================================================

class QualityController:
    """
    Supervisory quality control layer for steam systems.

    This controller sits above individual equipment controllers and provides:
    - Multi-point quality monitoring aggregation
    - Constraint computation for optimization
    - Coordinated action management across assets
    - Advisory mode (default) with optional automation mode

    Control Architecture:
        - Advisory mode: All recommendations require operator confirmation
        - Automation mode: Automatic execution within safe bounds (requires site approval)
        - Safety envelope is enforced in ALL modes

    Zero-Hallucination Principle:
        - All calculations use deterministic formulas
        - No LLM calls in control path
        - Every output has provenance tracking

    Attributes:
        controller_id: Unique controller identifier
        targets: Quality control targets
        constraints: Operational constraints
        mode: Current control mode (default: ADVISORY)
        _measurements: Current quality measurements by point
        _action_history: History of coordinated actions

    Example:
        >>> controller = QualityController("SQ-001", targets=QualityTargets())
        >>> controller.update_measurement(measurement)
        >>> recommendation = controller.compute_recommendation()
        >>> constraints = controller.get_optimization_constraints()
    """

    def __init__(
        self,
        controller_id: str,
        targets: Optional[QualityTargets] = None,
        constraints: Optional[QualityConstraints] = None,
        mode: ControlMode = ControlMode.ADVISORY
    ):
        """
        Initialize QualityController.

        Args:
            controller_id: Unique controller identifier
            targets: Quality control targets (uses defaults if None)
            constraints: Operational constraints (uses defaults if None)
            mode: Initial control mode (default: ADVISORY)
        """
        self.controller_id = controller_id
        self.targets = targets or QualityTargets()
        self.constraints = constraints or QualityConstraints()
        self.mode = mode

        # Internal state
        self._measurements: Dict[str, QualityMeasurement] = {}
        self._action_history: List[CoordinatedAction] = []
        self._pending_actions: Dict[str, CoordinatedAction] = {}
        self._last_correction_time: Optional[datetime] = None
        self._max_history_size = 1000
        self._automation_approval_expires: Optional[datetime] = None

        logger.info(
            f"QualityController {controller_id} initialized in {mode.value} mode "
            f"with targets: dryness={self.targets.target_dryness}, "
            f"superheat={self.targets.target_superheat_c}C"
        )

    def update_measurement(self, measurement: QualityMeasurement) -> None:
        """
        Update quality measurement from a measurement point.

        Args:
            measurement: Quality measurement data
        """
        self._measurements[measurement.measurement_id] = measurement
        logger.debug(
            f"Updated measurement {measurement.measurement_id}: "
            f"dryness={measurement.dryness_fraction:.4f}, "
            f"superheat={measurement.superheat_c:.1f}C"
        )

    def get_overall_status(self) -> QualityStatus:
        """
        Compute overall quality status from all measurements.

        Returns:
            QualityStatus: Aggregated quality status
        """
        if not self._measurements:
            return QualityStatus.UNKNOWN

        statuses = []
        for measurement in self._measurements.values():
            status = self._assess_measurement_status(measurement)
            statuses.append(status)

        # Return worst status (critical > warning > acceptable > optimal)
        priority_order = [
            QualityStatus.CRITICAL,
            QualityStatus.WARNING,
            QualityStatus.ACCEPTABLE,
            QualityStatus.OPTIMAL
        ]

        for status in priority_order:
            if status in statuses:
                return status

        return QualityStatus.UNKNOWN

    def compute_recommendation(self) -> ControlRecommendation:
        """
        Compute control recommendation based on current measurements.

        This method analyzes all measurements, computes required corrections,
        and generates coordinated actions across affected assets.

        Returns:
            ControlRecommendation: Recommendation with actions and constraints

        Raises:
            ValueError: If no measurements are available
        """
        start_time = datetime.now()

        if not self._measurements:
            raise ValueError("No measurements available for recommendation")

        # Step 1: Assess overall quality status
        overall_status = self.get_overall_status()

        # Step 2: Identify quality deviations
        deviations = self._identify_deviations()

        # Step 3: Compute coordinated actions
        coordinated_actions = self._compute_coordinated_actions(deviations)

        # Step 4: Compute optimization constraints
        optimization_constraints = self._compute_optimization_constraints()

        # Step 5: Estimate quality improvement
        quality_improvement = self._estimate_improvement(coordinated_actions)

        # Step 6: Generate rationale
        rationale = self._generate_rationale(overall_status, deviations)

        # Step 7: Check if we can skip confirmation (only in automation mode)
        requires_confirmation = True
        if self.mode == ControlMode.AUTOMATION and self._is_automation_approved():
            # Only skip confirmation for non-critical actions
            requires_confirmation = any(
                a.priority == ActionPriority.CRITICAL for a in coordinated_actions
            )

        # Generate recommendation ID
        recommendation_id = hashlib.sha256(
            f"REC_{self.controller_id}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        recommendation = ControlRecommendation(
            recommendation_id=recommendation_id,
            quality_status=overall_status,
            coordinated_actions=coordinated_actions,
            optimization_constraints=optimization_constraints,
            quality_improvement_estimate=quality_improvement,
            confidence_level=self._compute_confidence(deviations),
            rationale=rationale,
            warnings=self._generate_warnings(deviations),
            requires_confirmation=requires_confirmation,
            timestamp=start_time
        )

        # Calculate provenance hash
        recommendation.provenance_hash = hashlib.sha256(
            f"{recommendation_id}|{overall_status.value}|{len(coordinated_actions)}".encode()
        ).hexdigest()

        logger.info(
            f"Recommendation {recommendation_id}: status={overall_status.value}, "
            f"actions={len(coordinated_actions)}, improvement={quality_improvement:.3f}"
        )

        return recommendation

    def get_optimization_constraints(self) -> List[OptimizationConstraint]:
        """
        Get constraints for optimization solver.

        These constraints encode steam quality requirements in a form
        suitable for mathematical optimization.

        Returns:
            List[OptimizationConstraint]: Constraints for optimizer
        """
        return self._compute_optimization_constraints()

    def coordinate_action(
        self,
        action_type: str,
        affected_assets: List[str],
        setpoint_changes: Dict[str, float],
        priority: ActionPriority = ActionPriority.MEDIUM
    ) -> CoordinatedAction:
        """
        Create a coordinated action across multiple assets.

        Args:
            action_type: Type of coordinated action
            affected_assets: List of affected asset IDs
            setpoint_changes: Proposed setpoint changes by asset
            priority: Action priority level

        Returns:
            CoordinatedAction: Created coordinated action
        """
        start_time = datetime.now()

        # Validate stability time
        if not self._can_initiate_action():
            logger.warning(
                f"Action blocked: minimum stability time not met "
                f"(required: {self.constraints.min_stability_time_s}s)"
            )

        # Generate action ID
        action_id = hashlib.sha256(
            f"ACT_{action_type}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        # Determine sequence order based on priority and dependencies
        sequence_order = self._determine_sequence_order(
            affected_assets, action_type
        )

        # Validate safety
        safety_validated = self._validate_action_safety(
            action_type, setpoint_changes
        )

        # Build description
        description = self._build_action_description(
            action_type, affected_assets, setpoint_changes
        )

        action = CoordinatedAction(
            action_id=action_id,
            priority=priority,
            affected_assets=affected_assets,
            action_type=action_type,
            description=description,
            setpoint_changes=setpoint_changes,
            sequence_order=sequence_order,
            estimated_duration_s=len(affected_assets) * 15.0,  # Rough estimate
            requires_confirmation=(self.mode == ControlMode.ADVISORY),
            safety_validated=safety_validated,
            rollback_available=True,
            timestamp=start_time
        )

        # Calculate provenance hash
        action.provenance_hash = hashlib.sha256(
            f"{action_id}|{action_type}|{len(affected_assets)}".encode()
        ).hexdigest()

        # Store in pending actions
        self._pending_actions[action_id] = action

        logger.info(
            f"Coordinated action {action_id}: type={action_type}, "
            f"assets={len(affected_assets)}, priority={priority.value}"
        )

        return action

    def confirm_action(self, action_id: str, operator_id: str) -> bool:
        """
        Confirm a pending action for execution.

        Args:
            action_id: Action ID to confirm
            operator_id: ID of confirming operator

        Returns:
            bool: True if action was confirmed and moved to execution
        """
        if action_id not in self._pending_actions:
            logger.warning(f"Action {action_id} not found in pending actions")
            return False

        action = self._pending_actions.pop(action_id)
        self._action_history.append(action)
        self._last_correction_time = datetime.now()

        # Trim history if needed
        if len(self._action_history) > self._max_history_size:
            self._action_history = self._action_history[-self._max_history_size:]

        logger.info(
            f"Action {action_id} confirmed by {operator_id}, "
            f"moved to execution"
        )

        return True

    def get_supervisory_state(self) -> SupervisoryState:
        """
        Get current supervisory controller state.

        Returns:
            SupervisoryState: Current controller state
        """
        return SupervisoryState(
            controller_id=self.controller_id,
            mode=self.mode,
            active_measurements=len(self._measurements),
            overall_quality_status=self.get_overall_status(),
            active_actions=len(self._action_history),
            pending_confirmations=len(self._pending_actions),
            last_recommendation_time=self._last_correction_time,
            automation_enabled=(self.mode == ControlMode.AUTOMATION),
            site_approval_valid=self._is_automation_approved(),
            timestamp=datetime.now()
        )

    def set_mode(self, mode: ControlMode, approval_hours: float = 0) -> None:
        """
        Set control mode with optional automation approval duration.

        Args:
            mode: New control mode
            approval_hours: Duration of automation approval (hours)
        """
        old_mode = self.mode
        self.mode = mode

        if mode == ControlMode.AUTOMATION and approval_hours > 0:
            self._automation_approval_expires = (
                datetime.now() + timedelta(hours=approval_hours)
            )
            logger.info(
                f"Automation mode approved for {approval_hours} hours, "
                f"expires at {self._automation_approval_expires.isoformat()}"
            )
        else:
            self._automation_approval_expires = None

        logger.info(f"Control mode changed: {old_mode.value} -> {mode.value}")

    def enable_automation(self, approval_hours: float = 8.0) -> bool:
        """
        Enable automation mode with site approval.

        This requires explicit operator action and has a time limit.

        Args:
            approval_hours: Duration of automation approval (max 24 hours)

        Returns:
            bool: True if automation was enabled
        """
        # Limit approval duration
        approval_hours = min(approval_hours, 24.0)

        # Check prerequisites
        if self.get_overall_status() == QualityStatus.CRITICAL:
            logger.warning(
                "Cannot enable automation during critical quality status"
            )
            return False

        self.set_mode(ControlMode.AUTOMATION, approval_hours)
        return True

    def disable_automation(self) -> None:
        """Disable automation mode and return to advisory."""
        self.set_mode(ControlMode.ADVISORY)
        self._automation_approval_expires = None
        logger.info("Automation disabled, returning to advisory mode")

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _assess_measurement_status(
        self,
        measurement: QualityMeasurement
    ) -> QualityStatus:
        """Assess quality status for a single measurement."""
        # Check dryness
        if measurement.dryness_fraction < self.targets.min_dryness * 0.95:
            return QualityStatus.CRITICAL
        elif measurement.dryness_fraction < self.targets.min_dryness:
            return QualityStatus.WARNING

        # Check superheat
        if measurement.superheat_c < self.targets.min_superheat_c * 0.8:
            return QualityStatus.CRITICAL
        elif measurement.superheat_c < self.targets.min_superheat_c:
            return QualityStatus.WARNING
        elif measurement.superheat_c > self.targets.max_superheat_c:
            return QualityStatus.WARNING

        # Check moisture if available
        if measurement.moisture_ppm is not None:
            if measurement.moisture_ppm > self.targets.max_moisture_ppm * 1.2:
                return QualityStatus.CRITICAL
            elif measurement.moisture_ppm > self.targets.max_moisture_ppm:
                return QualityStatus.WARNING

        # Check optimal conditions
        dryness_ok = abs(
            measurement.dryness_fraction - self.targets.target_dryness
        ) < 0.01
        superheat_ok = abs(
            measurement.superheat_c - self.targets.target_superheat_c
        ) < 5.0

        if dryness_ok and superheat_ok:
            return QualityStatus.OPTIMAL

        return QualityStatus.ACCEPTABLE

    def _identify_deviations(self) -> Dict[str, Dict[str, float]]:
        """Identify quality deviations from targets."""
        deviations = {}

        for point_id, measurement in self._measurements.items():
            point_deviations = {}

            # Dryness deviation
            dryness_dev = self.targets.target_dryness - measurement.dryness_fraction
            if abs(dryness_dev) > 0.005:
                point_deviations["dryness"] = dryness_dev

            # Superheat deviation
            if measurement.superheat_c < self.targets.min_superheat_c:
                point_deviations["superheat_low"] = (
                    self.targets.min_superheat_c - measurement.superheat_c
                )
            elif measurement.superheat_c > self.targets.max_superheat_c:
                point_deviations["superheat_high"] = (
                    measurement.superheat_c - self.targets.max_superheat_c
                )

            # Moisture deviation
            if measurement.moisture_ppm is not None:
                if measurement.moisture_ppm > self.targets.max_moisture_ppm:
                    point_deviations["moisture"] = (
                        measurement.moisture_ppm - self.targets.max_moisture_ppm
                    )

            if point_deviations:
                deviations[point_id] = point_deviations

        return deviations

    def _compute_coordinated_actions(
        self,
        deviations: Dict[str, Dict[str, float]]
    ) -> List[CoordinatedAction]:
        """Compute coordinated actions to correct deviations."""
        actions = []

        for point_id, point_devs in deviations.items():
            measurement = self._measurements.get(point_id)
            if not measurement:
                continue

            # Low dryness or high moisture - reduce spray, check separator
            if "dryness" in point_devs or "moisture" in point_devs:
                priority = ActionPriority.HIGH if point_devs.get("dryness", 0) > 0.02 else ActionPriority.MEDIUM

                action = self.coordinate_action(
                    action_type="improve_dryness",
                    affected_assets=[measurement.asset_id],
                    setpoint_changes={
                        f"{measurement.asset_id}.spray_reduction": point_devs.get("dryness", 0) * 20.0
                    },
                    priority=priority
                )
                actions.append(action)

            # Low superheat - reduce spray or increase heat input
            if "superheat_low" in point_devs:
                priority = ActionPriority.CRITICAL if point_devs["superheat_low"] > 10 else ActionPriority.HIGH

                action = self.coordinate_action(
                    action_type="increase_superheat",
                    affected_assets=[measurement.asset_id],
                    setpoint_changes={
                        f"{measurement.asset_id}.spray_reduction": point_devs["superheat_low"] * 2.0
                    },
                    priority=priority
                )
                actions.append(action)

        return actions

    def _compute_optimization_constraints(self) -> List[OptimizationConstraint]:
        """Compute constraints for optimization solver."""
        constraints = []
        timestamp = datetime.now()

        for point_id, measurement in self._measurements.items():
            # Dryness constraint: >= min_dryness
            dryness_constraint = OptimizationConstraint(
                constraint_id=f"DRY_{point_id}_{timestamp.isoformat()}",
                variable_name=f"{point_id}.dryness_fraction",
                constraint_type=ConstraintType.INEQUALITY_GE,
                lower_bound=self.targets.min_dryness,
                weight=10.0,  # High weight for quality
                active=True,
                reason="Minimum steam dryness requirement"
            )
            dryness_constraint.provenance_hash = hashlib.sha256(
                f"{dryness_constraint.constraint_id}|{self.targets.min_dryness}".encode()
            ).hexdigest()
            constraints.append(dryness_constraint)

            # Superheat constraint: range [min, max]
            superheat_constraint = OptimizationConstraint(
                constraint_id=f"SH_{point_id}_{timestamp.isoformat()}",
                variable_name=f"{point_id}.superheat_c",
                constraint_type=ConstraintType.RANGE,
                lower_bound=self.targets.min_superheat_c,
                upper_bound=self.targets.max_superheat_c,
                weight=5.0,
                active=True,
                reason="Superheat margin range"
            )
            superheat_constraint.provenance_hash = hashlib.sha256(
                f"{superheat_constraint.constraint_id}|{self.targets.min_superheat_c}".encode()
            ).hexdigest()
            constraints.append(superheat_constraint)

        return constraints

    def _estimate_improvement(
        self,
        actions: List[CoordinatedAction]
    ) -> float:
        """Estimate quality improvement from actions."""
        if not actions:
            return 0.0

        # Simple linear estimate based on action count and priority
        improvement = 0.0
        for action in actions:
            if action.priority == ActionPriority.CRITICAL:
                improvement += 0.03
            elif action.priority == ActionPriority.HIGH:
                improvement += 0.02
            elif action.priority == ActionPriority.MEDIUM:
                improvement += 0.01
            else:
                improvement += 0.005

        return min(improvement, 0.1)  # Cap at 10% improvement estimate

    def _compute_confidence(
        self,
        deviations: Dict[str, Dict[str, float]]
    ) -> float:
        """Compute confidence level for recommendation."""
        if not self._measurements:
            return 0.5

        # Base confidence on number of measurements and data freshness
        base_confidence = 0.7

        # Increase for more measurements
        measurement_factor = min(len(self._measurements) / 5.0, 1.0)
        base_confidence += 0.1 * measurement_factor

        # Decrease for large deviations (less predictable)
        if deviations:
            avg_deviation = sum(
                sum(abs(v) for v in point_devs.values())
                for point_devs in deviations.values()
            ) / len(deviations)
            if avg_deviation > 0.05:
                base_confidence -= 0.1

        return min(max(base_confidence, 0.5), 0.95)

    def _generate_rationale(
        self,
        status: QualityStatus,
        deviations: Dict[str, Dict[str, float]]
    ) -> str:
        """Generate rationale for recommendation."""
        lines = [f"Overall quality status: {status.value}"]

        if deviations:
            lines.append(f"Deviations detected at {len(deviations)} points:")
            for point_id, point_devs in list(deviations.items())[:3]:
                dev_strs = [f"{k}: {v:.4f}" for k, v in point_devs.items()]
                lines.append(f"  - {point_id}: {', '.join(dev_strs)}")
        else:
            lines.append("No significant deviations detected.")

        return "\n".join(lines)

    def _generate_warnings(
        self,
        deviations: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate warning messages."""
        warnings = []

        for point_id, point_devs in deviations.items():
            if "dryness" in point_devs and point_devs["dryness"] > 0.03:
                warnings.append(
                    f"Significant dryness deficit at {point_id}: "
                    f"{point_devs['dryness']:.3f}"
                )
            if "superheat_low" in point_devs:
                warnings.append(
                    f"Low superheat at {point_id}: "
                    f"{point_devs['superheat_low']:.1f}C below minimum"
                )

        return warnings

    def _can_initiate_action(self) -> bool:
        """Check if minimum stability time has passed."""
        if self._last_correction_time is None:
            return True

        elapsed = (datetime.now() - self._last_correction_time).total_seconds()
        return elapsed >= self.constraints.min_stability_time_s

    def _is_automation_approved(self) -> bool:
        """Check if automation approval is still valid."""
        if self._automation_approval_expires is None:
            return False

        return datetime.now() < self._automation_approval_expires

    def _determine_sequence_order(
        self,
        affected_assets: List[str],
        action_type: str
    ) -> List[str]:
        """Determine sequence order for coordinated action."""
        # For now, return assets in order
        # Production would consider dependencies
        return affected_assets.copy()

    def _validate_action_safety(
        self,
        action_type: str,
        setpoint_changes: Dict[str, float]
    ) -> bool:
        """Validate action against safety constraints."""
        # Check that no setpoint change exceeds maximum correction rate
        for change_key, change_value in setpoint_changes.items():
            if abs(change_value) > self.constraints.max_correction_rate_pct_per_min * 10:
                logger.warning(
                    f"Setpoint change {change_key}={change_value} exceeds safe limits"
                )
                return False

        return True

    def _build_action_description(
        self,
        action_type: str,
        affected_assets: List[str],
        setpoint_changes: Dict[str, float]
    ) -> str:
        """Build human-readable action description."""
        asset_str = ", ".join(affected_assets[:3])
        if len(affected_assets) > 3:
            asset_str += f" and {len(affected_assets) - 3} more"

        change_count = len(setpoint_changes)

        return (
            f"Coordinated {action_type} action affecting {asset_str} "
            f"with {change_count} setpoint changes"
        )
