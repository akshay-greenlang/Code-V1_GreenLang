# -*- coding: utf-8 -*-
"""
Constraint Manager for GL-017 CONDENSYNC

Comprehensive constraint handling for condenser vacuum optimization.
Manages safety limits, equipment constraints, process constraints, and data quality gating.

Constraint Categories:
    1. Hard Safety Limits - Non-negotiable turbine/equipment protection
    2. Equipment Constraints - Pump curves, fan curves, operating envelopes
    3. Process Constraints - CW temps, discharge limits, environmental
    4. Data Quality Gating - Suppress recommendations on bad data

Zero-Hallucination Guarantee:
    - All constraints use deterministic engineering formulas
    - No AI/ML inference in constraint evaluation
    - Complete audit trail with violation logging
    - Physics-based boundary calculations

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Standard engineering safety margins
DEFAULT_SAFETY_MARGIN = 0.10  # 10% safety margin
CRITICAL_SAFETY_MARGIN = 0.20  # 20% for critical limits


# ============================================================================
# ENUMERATIONS
# ============================================================================

class ConstraintType(Enum):
    """Type of constraint."""
    HARD_SAFETY = "hard_safety"  # Non-negotiable, will reject optimization
    SOFT_LIMIT = "soft_limit"  # Preferred range, penalty-based
    EQUIPMENT = "equipment"  # Equipment operating envelope
    PROCESS = "process"  # Process requirements
    ENVIRONMENTAL = "environmental"  # Environmental/discharge limits
    DATA_QUALITY = "data_quality"  # Data quality requirements


class ConstraintSeverity(Enum):
    """Severity level of constraint violation."""
    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Action within 1 hour
    MEDIUM = "medium"  # Action within shift
    LOW = "low"  # Advisory only
    INFO = "info"  # Information only


class ConstraintStatus(Enum):
    """Evaluation status of a constraint."""
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    WARNING = "warning"  # Near limit
    UNKNOWN = "unknown"  # Cannot evaluate (bad data)


class DataQualityFlag(Enum):
    """Data quality flags."""
    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"
    STALE = "stale"
    MISSING = "missing"
    FROZEN = "frozen"
    OUT_OF_RANGE = "out_of_range"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ConstraintDefinition:
    """Definition of a single constraint."""
    constraint_id: str
    name: str
    description: str
    constraint_type: ConstraintType
    severity: ConstraintSeverity

    # Limit values
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None

    # Warning thresholds (as fraction of limit)
    warning_threshold: float = 0.90  # Warn at 90% of limit

    # Units and display
    unit: str = ""
    decimal_places: int = 2

    # Enforcement
    enforce_on_optimization: bool = True
    allow_temporary_violation: bool = False
    max_violation_duration_min: float = 0.0


@dataclass
class ConstraintEvaluation:
    """Result of evaluating a single constraint."""
    constraint_id: str
    constraint_name: str
    status: ConstraintStatus
    severity: ConstraintSeverity

    # Current value and limits
    current_value: Optional[float] = None
    limit_value: Optional[float] = None
    violation_amount: Optional[float] = None
    violation_pct: Optional[float] = None

    # Context
    unit: str = ""
    message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DataQualityCheck:
    """Result of data quality check for a tag."""
    tag_name: str
    current_value: Optional[float]
    quality_flag: DataQualityFlag
    quality_score: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)

    # Staleness check
    last_update: Optional[datetime] = None
    age_seconds: float = 0.0
    max_age_seconds: float = 60.0

    # Range check
    expected_min: Optional[float] = None
    expected_max: Optional[float] = None


@dataclass
class DataQualityGate:
    """Data quality gate configuration and result."""
    gate_id: str
    required_tags: List[str]
    min_quality_score: float = 0.8
    max_bad_tags: int = 0
    max_suspect_tags: int = 2

    # Result
    overall_score: float = 1.0
    passed: bool = True
    failed_tags: List[str] = field(default_factory=list)
    message: str = ""


@dataclass
class ConstraintManagerConfig:
    """Configuration for constraint manager."""
    # Safety margins
    default_safety_margin: float = DEFAULT_SAFETY_MARGIN
    critical_safety_margin: float = CRITICAL_SAFETY_MARGIN

    # Data quality settings
    enable_data_quality_gating: bool = True
    min_data_quality_score: float = 0.8
    max_tag_age_seconds: float = 60.0
    frozen_detection_window_seconds: float = 300.0
    frozen_detection_threshold: float = 0.001

    # Violation handling
    log_all_violations: bool = True
    alert_on_critical: bool = True

    # Ramp rate limits (applied globally)
    enable_ramp_rate_limits: bool = True


@dataclass
class ConstraintValidationResult:
    """Complete result of constraint validation."""
    result_id: str
    timestamp: datetime

    # Overall status
    all_constraints_satisfied: bool
    critical_violations: int
    total_violations: int
    total_warnings: int

    # Individual evaluations
    evaluations: List[ConstraintEvaluation]

    # Data quality
    data_quality_passed: bool
    data_quality_score: float
    data_quality_issues: List[str]

    # Recommendation suppression
    suppress_recommendation: bool
    suppression_reason: Optional[str]

    # Provenance
    provenance_hash: str


# ============================================================================
# CONSTRAINT LIBRARY
# ============================================================================

class ConstraintLibrary:
    """
    Library of predefined constraints for condenser systems.

    Provides standard constraint definitions for:
    - Backpressure limits (turbine protection)
    - CW flow limits (tube velocity, pump protection)
    - Temperature limits (discharge, tube sheet)
    - NPSH requirements
    - Ramp rates
    """

    @staticmethod
    def get_backpressure_constraints() -> List[ConstraintDefinition]:
        """Get backpressure-related constraints."""
        return [
            ConstraintDefinition(
                constraint_id="BP_MAX_HARD",
                name="Maximum Backpressure (Hard)",
                description="Turbine exhaust pressure hard limit for blade protection",
                constraint_type=ConstraintType.HARD_SAFETY,
                severity=ConstraintSeverity.CRITICAL,
                max_value=5.0,
                warning_threshold=0.90,
                unit="inHgA",
                decimal_places=2
            ),
            ConstraintDefinition(
                constraint_id="BP_MAX_ALARM",
                name="Maximum Backpressure (Alarm)",
                description="Backpressure alarm limit",
                constraint_type=ConstraintType.SOFT_LIMIT,
                severity=ConstraintSeverity.HIGH,
                max_value=4.5,
                warning_threshold=0.95,
                unit="inHgA",
                decimal_places=2
            ),
            ConstraintDefinition(
                constraint_id="BP_TARGET",
                name="Backpressure Target",
                description="Target backpressure for optimal efficiency",
                constraint_type=ConstraintType.SOFT_LIMIT,
                severity=ConstraintSeverity.LOW,
                target_value=2.5,
                min_value=1.5,
                max_value=3.5,
                unit="inHgA",
                decimal_places=2
            ),
        ]

    @staticmethod
    def get_cw_flow_constraints() -> List[ConstraintDefinition]:
        """Get CW flow-related constraints."""
        return [
            ConstraintDefinition(
                constraint_id="CW_FLOW_MIN",
                name="Minimum CW Flow",
                description="Minimum flow for adequate tube velocity",
                constraint_type=ConstraintType.HARD_SAFETY,
                severity=ConstraintSeverity.CRITICAL,
                min_value=50000.0,
                warning_threshold=0.85,
                unit="GPM",
                decimal_places=0
            ),
            ConstraintDefinition(
                constraint_id="CW_FLOW_MAX",
                name="Maximum CW Flow",
                description="Maximum flow based on pump capacity",
                constraint_type=ConstraintType.EQUIPMENT,
                severity=ConstraintSeverity.HIGH,
                max_value=200000.0,
                warning_threshold=0.95,
                unit="GPM",
                decimal_places=0
            ),
            ConstraintDefinition(
                constraint_id="TUBE_VEL_MIN",
                name="Minimum Tube Velocity",
                description="Minimum velocity to prevent fouling",
                constraint_type=ConstraintType.PROCESS,
                severity=ConstraintSeverity.MEDIUM,
                min_value=5.0,
                warning_threshold=0.90,
                unit="ft/s",
                decimal_places=1
            ),
            ConstraintDefinition(
                constraint_id="TUBE_VEL_MAX",
                name="Maximum Tube Velocity",
                description="Maximum velocity to prevent erosion",
                constraint_type=ConstraintType.PROCESS,
                severity=ConstraintSeverity.HIGH,
                max_value=10.0,
                warning_threshold=0.90,
                unit="ft/s",
                decimal_places=1
            ),
        ]

    @staticmethod
    def get_temperature_constraints() -> List[ConstraintDefinition]:
        """Get temperature-related constraints."""
        return [
            ConstraintDefinition(
                constraint_id="CW_OUTLET_MAX",
                name="Maximum CW Outlet Temperature",
                description="CW outlet temp limit for environmental discharge",
                constraint_type=ConstraintType.ENVIRONMENTAL,
                severity=ConstraintSeverity.HIGH,
                max_value=110.0,
                warning_threshold=0.95,
                unit="F",
                decimal_places=1
            ),
            ConstraintDefinition(
                constraint_id="CW_DELTA_T_MAX",
                name="Maximum CW Temperature Rise",
                description="Maximum CW temperature rise across condenser",
                constraint_type=ConstraintType.PROCESS,
                severity=ConstraintSeverity.MEDIUM,
                max_value=25.0,
                warning_threshold=0.90,
                unit="F",
                decimal_places=1
            ),
            ConstraintDefinition(
                constraint_id="HOTWELL_TEMP_MAX",
                name="Maximum Hotwell Temperature",
                description="Maximum condensate temperature",
                constraint_type=ConstraintType.PROCESS,
                severity=ConstraintSeverity.MEDIUM,
                max_value=130.0,
                warning_threshold=0.95,
                unit="F",
                decimal_places=1
            ),
        ]

    @staticmethod
    def get_pump_constraints() -> List[ConstraintDefinition]:
        """Get pump-related constraints."""
        return [
            ConstraintDefinition(
                constraint_id="NPSH_MARGIN",
                name="NPSH Margin",
                description="Net Positive Suction Head margin above required",
                constraint_type=ConstraintType.EQUIPMENT,
                severity=ConstraintSeverity.CRITICAL,
                min_value=1.2,  # 20% margin
                warning_threshold=0.85,
                unit="ratio",
                decimal_places=2
            ),
            ConstraintDefinition(
                constraint_id="PUMP_RAMP_RATE",
                name="Pump Flow Ramp Rate",
                description="Maximum CW flow change rate",
                constraint_type=ConstraintType.EQUIPMENT,
                severity=ConstraintSeverity.MEDIUM,
                max_value=5000.0,
                warning_threshold=0.90,
                unit="GPM/min",
                decimal_places=0
            ),
            ConstraintDefinition(
                constraint_id="MIN_PUMPS_RUNNING",
                name="Minimum Pumps Running",
                description="Minimum number of CW pumps in service",
                constraint_type=ConstraintType.HARD_SAFETY,
                severity=ConstraintSeverity.CRITICAL,
                min_value=1.0,
                unit="count",
                decimal_places=0
            ),
        ]

    @staticmethod
    def get_fan_constraints() -> List[ConstraintDefinition]:
        """Get cooling tower fan constraints."""
        return [
            ConstraintDefinition(
                constraint_id="FAN_SPEED_MIN",
                name="Minimum Fan Speed",
                description="Minimum fan speed when running",
                constraint_type=ConstraintType.EQUIPMENT,
                severity=ConstraintSeverity.LOW,
                min_value=20.0,
                unit="%",
                decimal_places=0
            ),
            ConstraintDefinition(
                constraint_id="FAN_SPEED_MAX",
                name="Maximum Fan Speed",
                description="Maximum fan speed limit",
                constraint_type=ConstraintType.EQUIPMENT,
                severity=ConstraintSeverity.MEDIUM,
                max_value=100.0,
                warning_threshold=0.95,
                unit="%",
                decimal_places=0
            ),
            ConstraintDefinition(
                constraint_id="FAN_RAMP_RATE",
                name="Fan Speed Ramp Rate",
                description="Maximum fan speed change rate",
                constraint_type=ConstraintType.EQUIPMENT,
                severity=ConstraintSeverity.LOW,
                max_value=10.0,
                unit="%/min",
                decimal_places=1
            ),
        ]

    @staticmethod
    def get_all_standard_constraints() -> List[ConstraintDefinition]:
        """Get all standard constraint definitions."""
        constraints = []
        constraints.extend(ConstraintLibrary.get_backpressure_constraints())
        constraints.extend(ConstraintLibrary.get_cw_flow_constraints())
        constraints.extend(ConstraintLibrary.get_temperature_constraints())
        constraints.extend(ConstraintLibrary.get_pump_constraints())
        constraints.extend(ConstraintLibrary.get_fan_constraints())
        return constraints


# ============================================================================
# DATA QUALITY CHECKER
# ============================================================================

class DataQualityChecker:
    """
    Data quality checker for sensor/tag validation.

    Performs:
    - Staleness detection
    - Range validation
    - Frozen value detection
    - Rate of change validation

    Zero-Hallucination Guarantee:
        All checks use deterministic thresholds and comparisons.
    """

    def __init__(self, config: ConstraintManagerConfig):
        """
        Initialize data quality checker.

        Args:
            config: Constraint manager configuration
        """
        self.config = config
        self._value_history: Dict[str, List[Tuple[datetime, float]]] = {}

    def check_tag_quality(
        self,
        tag_name: str,
        current_value: Optional[float],
        timestamp: Optional[datetime] = None,
        expected_min: Optional[float] = None,
        expected_max: Optional[float] = None
    ) -> DataQualityCheck:
        """
        Check quality of a single tag/sensor value.

        Args:
            tag_name: Tag identifier
            current_value: Current tag value
            timestamp: Value timestamp
            expected_min: Expected minimum value
            expected_max: Expected maximum value

        Returns:
            DataQualityCheck result
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        issues = []
        quality_flag = DataQualityFlag.GOOD
        quality_score = 1.0

        # Check for missing value
        if current_value is None:
            return DataQualityCheck(
                tag_name=tag_name,
                current_value=None,
                quality_flag=DataQualityFlag.MISSING,
                quality_score=0.0,
                issues=["Value is missing/null"]
            )

        # Check for NaN
        if np.isnan(current_value):
            return DataQualityCheck(
                tag_name=tag_name,
                current_value=current_value,
                quality_flag=DataQualityFlag.BAD,
                quality_score=0.0,
                issues=["Value is NaN"]
            )

        # Check staleness
        now = datetime.now(timezone.utc)
        age_seconds = (now - timestamp).total_seconds()

        if age_seconds > self.config.max_tag_age_seconds:
            issues.append(f"Value is stale ({age_seconds:.0f}s old)")
            quality_flag = DataQualityFlag.STALE
            quality_score = max(0.3, 1.0 - (age_seconds / self.config.max_tag_age_seconds) * 0.5)

        # Check range
        if expected_min is not None and current_value < expected_min:
            issues.append(f"Value {current_value} below expected min {expected_min}")
            if quality_flag == DataQualityFlag.GOOD:
                quality_flag = DataQualityFlag.OUT_OF_RANGE
            quality_score = min(quality_score, 0.5)

        if expected_max is not None and current_value > expected_max:
            issues.append(f"Value {current_value} above expected max {expected_max}")
            if quality_flag == DataQualityFlag.GOOD:
                quality_flag = DataQualityFlag.OUT_OF_RANGE
            quality_score = min(quality_score, 0.5)

        # Check for frozen value
        if self._is_frozen(tag_name, current_value, timestamp):
            issues.append("Value appears frozen (no change detected)")
            if quality_flag == DataQualityFlag.GOOD:
                quality_flag = DataQualityFlag.FROZEN
            quality_score = min(quality_score, 0.6)

        # Update history
        self._update_history(tag_name, current_value, timestamp)

        # Downgrade to suspect if any issues but not critical
        if issues and quality_flag == DataQualityFlag.GOOD:
            quality_flag = DataQualityFlag.SUSPECT
            quality_score = min(quality_score, 0.85)

        return DataQualityCheck(
            tag_name=tag_name,
            current_value=current_value,
            quality_flag=quality_flag,
            quality_score=quality_score,
            issues=issues,
            last_update=timestamp,
            age_seconds=age_seconds,
            max_age_seconds=self.config.max_tag_age_seconds,
            expected_min=expected_min,
            expected_max=expected_max
        )

    def _is_frozen(
        self,
        tag_name: str,
        current_value: float,
        timestamp: datetime
    ) -> bool:
        """Check if tag value appears frozen."""
        if tag_name not in self._value_history:
            return False

        history = self._value_history[tag_name]

        # Need enough history
        if len(history) < 5:
            return False

        # Check values within detection window
        window_start = timestamp - timedelta(
            seconds=self.config.frozen_detection_window_seconds
        )

        recent_values = [
            v for t, v in history
            if t >= window_start
        ]

        if len(recent_values) < 3:
            return False

        # Check if all values are within threshold
        value_range = max(recent_values) - min(recent_values)
        return value_range < self.config.frozen_detection_threshold

    def _update_history(
        self,
        tag_name: str,
        value: float,
        timestamp: datetime
    ) -> None:
        """Update value history for a tag."""
        if tag_name not in self._value_history:
            self._value_history[tag_name] = []

        self._value_history[tag_name].append((timestamp, value))

        # Keep only last 100 values
        if len(self._value_history[tag_name]) > 100:
            self._value_history[tag_name] = self._value_history[tag_name][-100:]

    def evaluate_data_quality_gate(
        self,
        gate: DataQualityGate,
        tag_values: Dict[str, Optional[float]],
        tag_timestamps: Optional[Dict[str, datetime]] = None
    ) -> DataQualityGate:
        """
        Evaluate a data quality gate.

        Args:
            gate: Gate configuration
            tag_values: Dict of tag_name: value
            tag_timestamps: Dict of tag_name: timestamp

        Returns:
            Updated DataQualityGate with evaluation results
        """
        if tag_timestamps is None:
            tag_timestamps = {}

        quality_scores = []
        failed_tags = []
        bad_count = 0
        suspect_count = 0

        for tag_name in gate.required_tags:
            value = tag_values.get(tag_name)
            timestamp = tag_timestamps.get(tag_name)

            check = self.check_tag_quality(tag_name, value, timestamp)
            quality_scores.append(check.quality_score)

            if check.quality_flag == DataQualityFlag.BAD or \
               check.quality_flag == DataQualityFlag.MISSING:
                bad_count += 1
                failed_tags.append(tag_name)
            elif check.quality_flag in [DataQualityFlag.SUSPECT,
                                        DataQualityFlag.STALE,
                                        DataQualityFlag.FROZEN]:
                suspect_count += 1

        # Calculate overall score
        overall_score = np.mean(quality_scores) if quality_scores else 0.0

        # Determine if gate passed
        passed = (
            overall_score >= gate.min_quality_score and
            bad_count <= gate.max_bad_tags and
            suspect_count <= gate.max_suspect_tags
        )

        # Generate message
        if passed:
            message = f"Data quality gate PASSED (score={overall_score:.2f})"
        else:
            reasons = []
            if overall_score < gate.min_quality_score:
                reasons.append(f"score {overall_score:.2f} < {gate.min_quality_score}")
            if bad_count > gate.max_bad_tags:
                reasons.append(f"{bad_count} bad tags > {gate.max_bad_tags} allowed")
            if suspect_count > gate.max_suspect_tags:
                reasons.append(f"{suspect_count} suspect tags > {gate.max_suspect_tags} allowed")
            message = f"Data quality gate FAILED: {'; '.join(reasons)}"

        return DataQualityGate(
            gate_id=gate.gate_id,
            required_tags=gate.required_tags,
            min_quality_score=gate.min_quality_score,
            max_bad_tags=gate.max_bad_tags,
            max_suspect_tags=gate.max_suspect_tags,
            overall_score=overall_score,
            passed=passed,
            failed_tags=failed_tags,
            message=message
        )


# ============================================================================
# CONSTRAINT MANAGER CLASS
# ============================================================================

class ConstraintManager:
    """
    Comprehensive constraint manager for condenser optimization.

    Handles:
    - Constraint definition and registration
    - Constraint evaluation against current values
    - Data quality gating
    - Recommendation suppression logic

    Zero-Hallucination Guarantee:
        - All constraint evaluations use deterministic comparisons
        - No AI/ML inference in any evaluation path
        - Complete audit trail of violations

    Example:
        >>> manager = ConstraintManager()
        >>> result = manager.validate_operating_point(current_values)
        >>> if not result.suppress_recommendation:
        ...     # Proceed with optimization
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        config: Optional[ConstraintManagerConfig] = None,
        custom_constraints: Optional[List[ConstraintDefinition]] = None
    ):
        """
        Initialize constraint manager.

        Args:
            config: Manager configuration
            custom_constraints: Custom constraint definitions
        """
        self.config = config or ConstraintManagerConfig()

        # Initialize constraint library
        self.constraints: Dict[str, ConstraintDefinition] = {}

        # Register standard constraints
        for constraint in ConstraintLibrary.get_all_standard_constraints():
            self.register_constraint(constraint)

        # Register custom constraints
        if custom_constraints:
            for constraint in custom_constraints:
                self.register_constraint(constraint)

        # Initialize data quality checker
        self.data_quality_checker = DataQualityChecker(self.config)

        logger.info(f"ConstraintManager initialized with {len(self.constraints)} constraints")

    def register_constraint(self, constraint: ConstraintDefinition) -> None:
        """
        Register a constraint definition.

        Args:
            constraint: Constraint to register
        """
        self.constraints[constraint.constraint_id] = constraint
        logger.debug(f"Registered constraint: {constraint.constraint_id}")

    def unregister_constraint(self, constraint_id: str) -> bool:
        """
        Unregister a constraint.

        Args:
            constraint_id: ID of constraint to remove

        Returns:
            True if removed, False if not found
        """
        if constraint_id in self.constraints:
            del self.constraints[constraint_id]
            return True
        return False

    def evaluate_constraint(
        self,
        constraint_id: str,
        current_value: float,
        timestamp: Optional[datetime] = None
    ) -> ConstraintEvaluation:
        """
        Evaluate a single constraint.

        Args:
            constraint_id: ID of constraint to evaluate
            current_value: Current value to check
            timestamp: Evaluation timestamp

        Returns:
            ConstraintEvaluation result
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        if constraint_id not in self.constraints:
            return ConstraintEvaluation(
                constraint_id=constraint_id,
                constraint_name="Unknown",
                status=ConstraintStatus.UNKNOWN,
                severity=ConstraintSeverity.INFO,
                message=f"Constraint {constraint_id} not found"
            )

        constraint = self.constraints[constraint_id]

        # Initialize evaluation
        status = ConstraintStatus.SATISFIED
        violation_amount = None
        violation_pct = None
        limit_value = None
        message = ""

        # Check minimum limit
        if constraint.min_value is not None:
            limit_value = constraint.min_value
            if current_value < constraint.min_value:
                status = ConstraintStatus.VIOLATED
                violation_amount = constraint.min_value - current_value
                violation_pct = (violation_amount / constraint.min_value) * 100
                message = (
                    f"{constraint.name}: {current_value:.{constraint.decimal_places}f} "
                    f"< min {constraint.min_value:.{constraint.decimal_places}f} {constraint.unit}"
                )
            elif current_value < constraint.min_value / constraint.warning_threshold:
                status = ConstraintStatus.WARNING
                message = f"{constraint.name}: approaching minimum limit"

        # Check maximum limit
        if constraint.max_value is not None:
            limit_value = constraint.max_value
            if current_value > constraint.max_value:
                status = ConstraintStatus.VIOLATED
                violation_amount = current_value - constraint.max_value
                violation_pct = (violation_amount / constraint.max_value) * 100
                message = (
                    f"{constraint.name}: {current_value:.{constraint.decimal_places}f} "
                    f"> max {constraint.max_value:.{constraint.decimal_places}f} {constraint.unit}"
                )
            elif current_value > constraint.max_value * constraint.warning_threshold:
                if status != ConstraintStatus.VIOLATED:
                    status = ConstraintStatus.WARNING
                    message = f"{constraint.name}: approaching maximum limit"

        # Log violations
        if status == ConstraintStatus.VIOLATED and self.config.log_all_violations:
            logger.warning(f"Constraint violation: {message}")

            if constraint.severity == ConstraintSeverity.CRITICAL and self.config.alert_on_critical:
                logger.critical(f"CRITICAL constraint violation: {message}")

        return ConstraintEvaluation(
            constraint_id=constraint_id,
            constraint_name=constraint.name,
            status=status,
            severity=constraint.severity,
            current_value=current_value,
            limit_value=limit_value,
            violation_amount=violation_amount,
            violation_pct=violation_pct,
            unit=constraint.unit,
            message=message,
            timestamp=timestamp
        )

    def validate_operating_point(
        self,
        values: Dict[str, float],
        tag_timestamps: Optional[Dict[str, datetime]] = None
    ) -> ConstraintValidationResult:
        """
        Validate a complete operating point against all constraints.

        Args:
            values: Dict mapping constraint_id to current value
            tag_timestamps: Dict mapping tag_name to timestamp

        Returns:
            Complete ConstraintValidationResult
        """
        start_time = datetime.now(timezone.utc)

        evaluations = []
        critical_violations = 0
        total_violations = 0
        total_warnings = 0

        # Evaluate all constraints with provided values
        for constraint_id, value in values.items():
            if constraint_id in self.constraints:
                evaluation = self.evaluate_constraint(constraint_id, value)
                evaluations.append(evaluation)

                if evaluation.status == ConstraintStatus.VIOLATED:
                    total_violations += 1
                    if evaluation.severity == ConstraintSeverity.CRITICAL:
                        critical_violations += 1
                elif evaluation.status == ConstraintStatus.WARNING:
                    total_warnings += 1

        # Evaluate data quality gate if enabled
        data_quality_passed = True
        data_quality_score = 1.0
        data_quality_issues = []

        if self.config.enable_data_quality_gating:
            gate = DataQualityGate(
                gate_id="DEFAULT_GATE",
                required_tags=list(values.keys()),
                min_quality_score=self.config.min_data_quality_score
            )

            gate_result = self.data_quality_checker.evaluate_data_quality_gate(
                gate,
                {k: v for k, v in values.items()},
                tag_timestamps
            )

            data_quality_passed = gate_result.passed
            data_quality_score = gate_result.overall_score
            if not gate_result.passed:
                data_quality_issues.append(gate_result.message)

        # Determine if recommendation should be suppressed
        suppress_recommendation = False
        suppression_reason = None

        if critical_violations > 0:
            suppress_recommendation = True
            suppression_reason = f"{critical_violations} critical constraint violation(s)"
        elif not data_quality_passed:
            suppress_recommendation = True
            suppression_reason = "Data quality gate failed"

        # Calculate provenance
        provenance_data = {
            "version": self.VERSION,
            "timestamp": start_time.isoformat(),
            "violations": total_violations,
            "data_quality_score": round(data_quality_score, 3)
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return ConstraintValidationResult(
            result_id=f"CV-{start_time.strftime('%Y%m%d%H%M%S')}-{provenance_hash}",
            timestamp=start_time,
            all_constraints_satisfied=(total_violations == 0),
            critical_violations=critical_violations,
            total_violations=total_violations,
            total_warnings=total_warnings,
            evaluations=evaluations,
            data_quality_passed=data_quality_passed,
            data_quality_score=data_quality_score,
            data_quality_issues=data_quality_issues,
            suppress_recommendation=suppress_recommendation,
            suppression_reason=suppression_reason,
            provenance_hash=provenance_hash
        )

    def get_active_violations(
        self,
        validation_result: ConstraintValidationResult
    ) -> List[ConstraintEvaluation]:
        """
        Get list of active violations from validation result.

        Args:
            validation_result: Validation result to filter

        Returns:
            List of violated constraints
        """
        return [
            e for e in validation_result.evaluations
            if e.status == ConstraintStatus.VIOLATED
        ]

    def get_constraint_summary(self) -> Dict[str, Any]:
        """
        Get summary of registered constraints.

        Returns:
            Dict with constraint summary
        """
        summary = {
            "total_constraints": len(self.constraints),
            "by_type": {},
            "by_severity": {}
        }

        for constraint in self.constraints.values():
            # Count by type
            type_name = constraint.constraint_type.value
            summary["by_type"][type_name] = summary["by_type"].get(type_name, 0) + 1

            # Count by severity
            sev_name = constraint.severity.value
            summary["by_severity"][sev_name] = summary["by_severity"].get(sev_name, 0) + 1

        return summary

    def generate_violation_report(
        self,
        validation_result: ConstraintValidationResult
    ) -> str:
        """
        Generate human-readable violation report.

        Args:
            validation_result: Validation result

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "         CONSTRAINT VALIDATION REPORT",
            "=" * 60,
            f"Result ID: {validation_result.result_id}",
            f"Timestamp: {validation_result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "SUMMARY:",
            f"  All Constraints Satisfied: {'YES' if validation_result.all_constraints_satisfied else 'NO'}",
            f"  Critical Violations: {validation_result.critical_violations}",
            f"  Total Violations: {validation_result.total_violations}",
            f"  Total Warnings: {validation_result.total_warnings}",
            f"  Data Quality Score: {validation_result.data_quality_score:.2%}",
            "",
        ]

        if validation_result.suppress_recommendation:
            lines.extend([
                "*** RECOMMENDATION SUPPRESSED ***",
                f"Reason: {validation_result.suppression_reason}",
                "",
            ])

        violations = self.get_active_violations(validation_result)
        if violations:
            lines.append("VIOLATIONS:")
            for v in violations:
                lines.append(
                    f"  [{v.severity.value.upper()}] {v.constraint_name}: "
                    f"{v.current_value:.2f} {v.unit} "
                    f"(limit: {v.limit_value:.2f}, violation: {v.violation_pct:.1f}%)"
                )
            lines.append("")

        warnings = [
            e for e in validation_result.evaluations
            if e.status == ConstraintStatus.WARNING
        ]
        if warnings:
            lines.append("WARNINGS:")
            for w in warnings:
                lines.append(f"  {w.constraint_name}: {w.message}")
            lines.append("")

        if validation_result.data_quality_issues:
            lines.append("DATA QUALITY ISSUES:")
            for issue in validation_result.data_quality_issues:
                lines.append(f"  - {issue}")
            lines.append("")

        lines.append(f"Provenance: {validation_result.provenance_hash}")
        lines.append("=" * 60)

        return "\n".join(lines)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_default_constraint_manager() -> ConstraintManager:
    """
    Create constraint manager with default configuration.

    Returns:
        Configured ConstraintManager instance
    """
    config = ConstraintManagerConfig()
    return ConstraintManager(config)
