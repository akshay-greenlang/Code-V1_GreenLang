# -*- coding: utf-8 -*-
"""
GL-004 Burnmaster - Uncertainty Gate Module

Implements uncertainty-based gating for combustion control recommendations.
Prevents recommendations when uncertainty is too high to ensure safe,
reliable operation.

Key Principles:
    - Block recommendations when uncertainty exceeds thresholds
    - Log all gate decisions for audit trails
    - Recommend actions to reduce uncertainty
    - Support configurable thresholds per variable type

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import numpy as np
import hashlib
import json
import logging


# Configure logger
logger = logging.getLogger(__name__)


class GateStatus(str, Enum):
    """Status of uncertainty gate check."""
    PASS = "pass"  # Uncertainty within threshold
    WARN = "warn"  # Uncertainty elevated but acceptable
    BLOCK = "block"  # Uncertainty too high, block recommendation
    ERROR = "error"  # Unable to evaluate


class ActionType(str, Enum):
    """Types of recommended actions to reduce uncertainty."""
    CALIBRATE_SENSOR = "calibrate_sensor"
    INCREASE_SAMPLING = "increase_sampling"
    CHECK_SENSOR_DRIFT = "check_sensor_drift"
    WAIT_FOR_STEADY_STATE = "wait_for_steady_state"
    VERIFY_MEASUREMENT = "verify_measurement"
    MANUAL_INTERVENTION = "manual_intervention"
    NO_ACTION = "no_action"


class VariableCategory(str, Enum):
    """Categories of combustion variables with different thresholds."""
    EMISSION = "emission"  # NOx, CO, CO2 - strict thresholds
    EFFICIENCY = "efficiency"  # Combustion efficiency - moderate
    SAFETY = "safety"  # O2, flame detection - very strict
    CONTROL = "control"  # Air/fuel ratio, damper positions - moderate
    DIAGNOSTIC = "diagnostic"  # Monitoring only - relaxed


@dataclass
class GateResult:
    """
    Result of an uncertainty gate check.

    Attributes:
        status: Gate status (pass/warn/block/error)
        uncertainty: Actual uncertainty value
        threshold: Threshold used for comparison
        margin: How far from threshold (positive = within, negative = exceeded)
        message: Human-readable status message
        variable_name: Name of variable checked
        timestamp: When check was performed
    """
    status: GateStatus
    uncertainty: float
    threshold: float
    margin: float = field(init=False)
    message: str = ""
    variable_name: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: str = ""

    def __post_init__(self):
        self.margin = self.threshold - self.uncertainty
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "status": self.status.value,
            "uncertainty": self.uncertainty,
            "threshold": self.threshold,
            "margin": self.margin,
            "variable_name": self.variable_name,
            "timestamp": self.timestamp.isoformat(),
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class BlockResult:
    """
    Result of blocking decision on high uncertainty.

    Attributes:
        is_blocked: Whether the recommendation is blocked
        value: The value being evaluated
        uncertainty: Uncertainty of the value
        relative_uncertainty: Uncertainty as percentage of value
        max_allowed: Maximum allowed relative uncertainty
        reason: Reason for blocking (if blocked)
        alternative_action: Suggested alternative when blocked
    """
    is_blocked: bool
    value: float
    uncertainty: float
    relative_uncertainty: float = field(init=False)
    max_allowed: float = 5.0
    reason: str = ""
    alternative_action: str = ""
    provenance_hash: str = ""

    def __post_init__(self):
        self.relative_uncertainty = (
            (self.uncertainty / abs(self.value)) * 100
            if self.value != 0 else float('inf')
        )
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "is_blocked": self.is_blocked,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "relative_uncertainty": self.relative_uncertainty,
            "max_allowed": self.max_allowed,
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class ActionRecommendation:
    """
    Recommended action to reduce uncertainty.

    Attributes:
        action_type: Type of recommended action
        priority: Priority level (1=highest, 5=lowest)
        description: Detailed description of recommended action
        expected_improvement: Expected uncertainty reduction (%)
        estimated_time: Estimated time to implement (minutes)
        affected_sensors: Sensors involved in the action
    """
    action_type: ActionType
    priority: int = 3
    description: str = ""
    expected_improvement: float = 0.0
    estimated_time: float = 0.0
    affected_sensors: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def __post_init__(self):
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "action_type": self.action_type.value,
            "priority": self.priority,
            "description": self.description,
            "expected_improvement": self.expected_improvement,
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class UncertaintyReport:
    """
    Comprehensive uncertainty report for gate analysis.

    Attributes:
        variable_name: Name of the variable
        value: Measured/calculated value
        standard_uncertainty: Standard uncertainty
        expanded_uncertainty: Expanded uncertainty (k=2)
        relative_uncertainty_percent: Uncertainty as percentage
        category: Variable category
        dominant_contributors: Top contributors to uncertainty
        is_acceptable: Whether uncertainty is acceptable
    """
    variable_name: str
    value: float
    standard_uncertainty: float
    expanded_uncertainty: float
    relative_uncertainty_percent: float
    category: VariableCategory
    dominant_contributors: List[str] = field(default_factory=list)
    is_acceptable: bool = True


@dataclass
class GateDecision:
    """
    Complete gate decision record for logging.

    Attributes:
        decision_id: Unique decision identifier
        timestamp: When decision was made
        variable_name: Variable being evaluated
        gate_result: Result of gate check
        block_result: Result of block evaluation
        recommendations: Recommended actions
        context: Additional context data
    """
    decision_id: str
    timestamp: datetime
    variable_name: str
    gate_result: GateResult
    block_result: Optional[BlockResult] = None
    recommendations: List[ActionRecommendation] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

    def __post_init__(self):
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "variable_name": self.variable_name,
            "gate_status": self.gate_result.status.value,
            "gate_uncertainty": self.gate_result.uncertainty,
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


class UncertaintyGate:
    """
    Uncertainty-based gating for combustion control recommendations.

    Prevents recommendations when uncertainty is too high to ensure
    safe, reliable combustion optimization. All decisions are logged
    for audit and compliance.

    ZERO HALLUCINATION: All thresholds are explicit and deterministic.
    No LLM inference in gate decisions.

    Example Usage:
        >>> gate = UncertaintyGate()
        >>> result = gate.check_uncertainty_threshold(uncertainty=0.5, threshold=1.0)
        >>> if result.status == GateStatus.PASS:
        ...     print("Recommendation allowed")
        >>> else:
        ...     print(f"Blocked: {result.message}")
    """

    # Default thresholds by variable category (relative uncertainty %)
    DEFAULT_THRESHOLDS: Dict[VariableCategory, Dict[str, float]] = {
        VariableCategory.SAFETY: {
            "pass": 2.0,  # <2% relative uncertainty OK
            "warn": 3.0,  # 2-3% warning
            "block": 5.0,  # >5% blocked
        },
        VariableCategory.EMISSION: {
            "pass": 5.0,
            "warn": 10.0,
            "block": 15.0,
        },
        VariableCategory.EFFICIENCY: {
            "pass": 3.0,
            "warn": 5.0,
            "block": 10.0,
        },
        VariableCategory.CONTROL: {
            "pass": 5.0,
            "warn": 10.0,
            "block": 20.0,
        },
        VariableCategory.DIAGNOSTIC: {
            "pass": 10.0,
            "warn": 20.0,
            "block": 50.0,
        },
    }

    def __init__(
        self,
        custom_thresholds: Optional[Dict[VariableCategory, Dict[str, float]]] = None,
        log_decisions: bool = True,
    ):
        """
        Initialize the uncertainty gate.

        Args:
            custom_thresholds: Override default thresholds
            log_decisions: Whether to log all decisions
        """
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)

        self.log_decisions = log_decisions
        self._decision_history: List[GateDecision] = []
        self._decision_counter = 0

    def check_uncertainty_threshold(
        self,
        uncertainty: float,
        threshold: float,
        variable_name: str = "unknown",
    ) -> GateResult:
        """
        Check if uncertainty is within acceptable threshold.

        Simple threshold comparison with status assignment.

        Args:
            uncertainty: Uncertainty value to check
            threshold: Maximum acceptable uncertainty
            variable_name: Name of variable (for logging)

        Returns:
            GateResult with pass/warn/block status

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        if uncertainty <= threshold * 0.8:
            status = GateStatus.PASS
            message = f"Uncertainty {uncertainty:.4f} well within threshold {threshold:.4f}"
        elif uncertainty <= threshold:
            status = GateStatus.WARN
            message = f"Uncertainty {uncertainty:.4f} approaching threshold {threshold:.4f}"
        else:
            status = GateStatus.BLOCK
            message = f"Uncertainty {uncertainty:.4f} exceeds threshold {threshold:.4f}"

        return GateResult(
            status=status,
            uncertainty=uncertainty,
            threshold=threshold,
            message=message,
            variable_name=variable_name,
        )

    def block_on_high_uncertainty(
        self,
        value: float,
        uncertainty: float,
        max_allowed: float = 5.0,
        variable_name: str = "unknown",
    ) -> BlockResult:
        """
        Determine if recommendation should be blocked due to high uncertainty.

        Uses relative uncertainty (uncertainty/value) for comparison.

        Args:
            value: The measured/calculated value
            uncertainty: Standard or expanded uncertainty
            max_allowed: Maximum allowed relative uncertainty (%)
            variable_name: Name of variable (for logging)

        Returns:
            BlockResult indicating whether to block recommendation

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        # Calculate relative uncertainty
        if abs(value) > 1e-10:
            relative_unc = (uncertainty / abs(value)) * 100
        else:
            # Value near zero - use absolute threshold
            relative_unc = float('inf') if uncertainty > 0 else 0

        is_blocked = relative_unc > max_allowed

        if is_blocked:
            reason = (
                f"Relative uncertainty {relative_unc:.1f}% exceeds "
                f"maximum allowed {max_allowed:.1f}%"
            )
            alternative = (
                "Wait for improved measurements or perform sensor calibration "
                "before acting on this recommendation"
            )
        else:
            reason = ""
            alternative = ""

        return BlockResult(
            is_blocked=is_blocked,
            value=value,
            uncertainty=uncertainty,
            max_allowed=max_allowed,
            reason=reason,
            alternative_action=alternative,
        )

    def recommend_action_on_uncertainty(
        self,
        uncertainty: UncertaintyReport,
    ) -> ActionRecommendation:
        """
        Recommend action to reduce uncertainty based on report.

        Analyzes uncertainty contributors and suggests appropriate
        actions to improve measurement quality.

        Args:
            uncertainty: Comprehensive uncertainty report

        Returns:
            ActionRecommendation with suggested improvement action

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        # Get threshold for this category
        thresholds = self.thresholds.get(
            uncertainty.category,
            self.DEFAULT_THRESHOLDS[VariableCategory.CONTROL]
        )

        rel_unc = uncertainty.relative_uncertainty_percent

        # Determine action based on uncertainty level and contributors
        if rel_unc <= thresholds["pass"]:
            return ActionRecommendation(
                action_type=ActionType.NO_ACTION,
                priority=5,
                description="Uncertainty is within acceptable limits",
                expected_improvement=0.0,
                estimated_time=0.0,
            )

        # Analyze dominant contributors
        contributors = uncertainty.dominant_contributors

        if "drift" in " ".join(contributors).lower():
            return ActionRecommendation(
                action_type=ActionType.CALIBRATE_SENSOR,
                priority=1,
                description=(
                    f"High drift detected in {uncertainty.variable_name}. "
                    "Recommend sensor calibration to reduce uncertainty."
                ),
                expected_improvement=50.0,
                estimated_time=30.0,
                affected_sensors=contributors,
            )

        if "noise" in " ".join(contributors).lower() or "random" in " ".join(contributors).lower():
            return ActionRecommendation(
                action_type=ActionType.INCREASE_SAMPLING,
                priority=2,
                description=(
                    f"High measurement noise in {uncertainty.variable_name}. "
                    "Recommend increasing sampling rate or averaging period."
                ),
                expected_improvement=30.0,
                estimated_time=5.0,
            )

        if rel_unc > thresholds["block"]:
            return ActionRecommendation(
                action_type=ActionType.MANUAL_INTERVENTION,
                priority=1,
                description=(
                    f"Very high uncertainty ({rel_unc:.1f}%) in {uncertainty.variable_name}. "
                    "Manual verification recommended before automation."
                ),
                expected_improvement=0.0,
                estimated_time=60.0,
            )

        # Default action for elevated uncertainty
        return ActionRecommendation(
            action_type=ActionType.VERIFY_MEASUREMENT,
            priority=3,
            description=(
                f"Elevated uncertainty ({rel_unc:.1f}%) in {uncertainty.variable_name}. "
                "Verify sensor readings and check for process disturbances."
            ),
            expected_improvement=20.0,
            estimated_time=10.0,
        )

    def log_uncertainty_gate_decision(
        self,
        decision: GateDecision,
    ) -> None:
        """
        Log a gate decision for audit trail.

        Stores decision in history and logs to system logger.

        Args:
            decision: Complete gate decision record
        """
        if not self.log_decisions:
            return

        # Store in history
        self._decision_history.append(decision)

        # Log to system logger
        log_msg = (
            f"UncertaintyGate Decision [{decision.decision_id}]: "
            f"Variable={decision.variable_name}, "
            f"Status={decision.gate_result.status.value}, "
            f"Uncertainty={decision.gate_result.uncertainty:.4f}, "
            f"Threshold={decision.gate_result.threshold:.4f}"
        )

        if decision.gate_result.status == GateStatus.BLOCK:
            logger.warning(log_msg)
        elif decision.gate_result.status == GateStatus.WARN:
            logger.info(log_msg)
        else:
            logger.debug(log_msg)

    def evaluate_and_gate(
        self,
        value: float,
        uncertainty: float,
        variable_name: str,
        category: VariableCategory = VariableCategory.CONTROL,
        context: Optional[Dict[str, Any]] = None,
    ) -> GateDecision:
        """
        Complete evaluation and gating workflow.

        Performs threshold check, block evaluation, and generates
        recommendations in one call.

        Args:
            value: The measured/calculated value
            uncertainty: Standard uncertainty
            variable_name: Name of the variable
            category: Variable category for threshold selection
            context: Additional context for logging

        Returns:
            Complete GateDecision with all components

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        self._decision_counter += 1
        decision_id = f"GD-{self._decision_counter:06d}"

        # Get thresholds for category
        thresholds = self.thresholds.get(category, self.DEFAULT_THRESHOLDS[category])

        # Calculate relative uncertainty
        if abs(value) > 1e-10:
            rel_unc = (uncertainty / abs(value)) * 100
        else:
            rel_unc = float('inf') if uncertainty > 0 else 0

        # Check against threshold
        gate_result = self.check_uncertainty_threshold(
            uncertainty=rel_unc,
            threshold=thresholds["block"],
            variable_name=variable_name,
        )

        # Block evaluation
        block_result = self.block_on_high_uncertainty(
            value=value,
            uncertainty=uncertainty,
            max_allowed=thresholds["block"],
            variable_name=variable_name,
        )

        # Generate recommendation
        report = UncertaintyReport(
            variable_name=variable_name,
            value=value,
            standard_uncertainty=uncertainty,
            expanded_uncertainty=2.0 * uncertainty,
            relative_uncertainty_percent=rel_unc,
            category=category,
            is_acceptable=(gate_result.status != GateStatus.BLOCK),
        )
        recommendation = self.recommend_action_on_uncertainty(report)

        # Create decision record
        decision = GateDecision(
            decision_id=decision_id,
            timestamp=datetime.utcnow(),
            variable_name=variable_name,
            gate_result=gate_result,
            block_result=block_result,
            recommendations=[recommendation],
            context=context or {},
        )

        # Log decision
        self.log_uncertainty_gate_decision(decision)

        return decision

    def get_decision_history(
        self,
        variable_name: Optional[str] = None,
        status: Optional[GateStatus] = None,
        since: Optional[datetime] = None,
    ) -> List[GateDecision]:
        """
        Retrieve decision history with optional filters.

        Args:
            variable_name: Filter by variable name
            status: Filter by gate status
            since: Filter to decisions after this time

        Returns:
            List of matching GateDecision records
        """
        results = self._decision_history

        if variable_name:
            results = [d for d in results if d.variable_name == variable_name]

        if status:
            results = [d for d in results if d.gate_result.status == status]

        if since:
            results = [d for d in results if d.timestamp >= since]

        return results

    def get_block_rate(
        self,
        variable_name: Optional[str] = None,
    ) -> float:
        """
        Calculate the rate of blocked decisions.

        Args:
            variable_name: Optional filter by variable

        Returns:
            Fraction of decisions that were blocked (0.0 to 1.0)
        """
        decisions = self.get_decision_history(variable_name=variable_name)

        if not decisions:
            return 0.0

        blocked = sum(
            1 for d in decisions
            if d.gate_result.status == GateStatus.BLOCK
        )

        return blocked / len(decisions)

    def clear_history(self) -> None:
        """Clear decision history."""
        self._decision_history.clear()
