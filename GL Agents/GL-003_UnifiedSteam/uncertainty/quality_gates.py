"""
Uncertainty-Based Quality Gates for GL-003 UNIFIEDSTEAM SteamSystemOptimizer.

This module implements gating logic that prevents aggressive recommendations
when uncertainty is too high, and requires operator confirmation for
high-risk actions with uncertain outcomes.

Zero-Hallucination Guarantee:
- All gate decisions are based on deterministic threshold comparisons
- No LLM inference in gating logic
- Complete audit trail for all gate decisions
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import logging

from .uncertainty_models import (
    UncertainValue,
    PropagatedUncertainty,
    PropertyUncertainty,
    ConfidenceLevel
)


logger = logging.getLogger(__name__)


class GateStatus(Enum):
    """Possible outcomes of a quality gate check."""
    PASSED = "passed"              # Uncertainty acceptable, proceed
    WARNING = "warning"            # Elevated uncertainty, proceed with caution
    BLOCKED = "blocked"            # Uncertainty too high, action blocked
    REQUIRES_CONFIRMATION = "requires_confirmation"  # Need operator approval


class RiskLevel(Enum):
    """Risk level classification for recommendations."""
    LOW = "low"           # Minor adjustments, easily reversible
    MEDIUM = "medium"     # Significant changes, reversible with effort
    HIGH = "high"         # Major changes, difficult to reverse
    CRITICAL = "critical" # Safety-critical, potentially irreversible


class WarningPriority(Enum):
    """Priority levels for uncertainty warnings."""
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


@dataclass
class GateResult:
    """
    Result of a quality gate check.

    Attributes:
        status: Gate status (passed, warning, blocked, requires_confirmation)
        recommendation_id: ID of the recommendation being gated
        uncertainty_level: Measured uncertainty level
        threshold: Threshold that was compared against
        confidence_level: Confidence level of the uncertainty
        reason: Human-readable explanation
        required_action: What action is needed (if any)
        timestamp: When the gate check was performed
        provenance_hash: Audit trail hash
    """
    status: GateStatus
    recommendation_id: str
    uncertainty_level: float
    threshold: float
    confidence_level: float
    reason: str
    required_action: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: str = ""

    def __post_init__(self):
        """Compute provenance hash."""
        if not self.provenance_hash:
            hash_data = {
                "status": self.status.value,
                "recommendation_id": self.recommendation_id,
                "uncertainty_level": self.uncertainty_level,
                "threshold": self.threshold,
                "timestamp": self.timestamp.isoformat()
            }
            hash_str = json.dumps(hash_data, sort_keys=True)
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()

    def is_blocked(self) -> bool:
        """Check if the recommendation is blocked."""
        return self.status == GateStatus.BLOCKED

    def requires_human_action(self) -> bool:
        """Check if human action is required."""
        return self.status in [GateStatus.BLOCKED, GateStatus.REQUIRES_CONFIRMATION]


@dataclass
class QualityCheckResult:
    """
    Result of measurement quality check.

    Attributes:
        overall_status: Overall quality status
        checked_measurements: Number of measurements checked
        passed_count: Number that passed quality check
        warning_count: Number with warnings
        failed_count: Number that failed
        details: Per-measurement details
        worst_measurement: ID of measurement with highest uncertainty
        recommendations: Quality improvement recommendations
    """
    overall_status: GateStatus
    checked_measurements: int
    passed_count: int
    warning_count: int
    failed_count: int
    details: Dict[str, Dict[str, Any]]
    worst_measurement: str
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Warning:
    """
    Uncertainty warning for display in UI.

    Attributes:
        warning_id: Unique warning identifier
        priority: Warning priority level
        title: Short warning title
        message: Detailed warning message
        affected_sensors: Sensors contributing to uncertainty
        affected_calculations: Calculations affected
        recommended_actions: Suggested actions to reduce uncertainty
        suppress_until: Optional time to suppress warning until
    """
    warning_id: str
    priority: WarningPriority
    title: str
    message: str
    affected_sensors: List[str]
    affected_calculations: List[str]
    recommended_actions: List[str]
    suppress_until: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Recommendation:
    """
    System recommendation that may be gated by uncertainty.

    Attributes:
        recommendation_id: Unique identifier
        action_type: Type of action (setpoint_change, mode_change, etc.)
        target: Target component or parameter
        current_value: Current value
        proposed_value: Proposed new value
        expected_benefit: Expected benefit/savings
        uncertainty: Uncertainty in expected benefit
        risk_level: Risk classification
        confidence: Confidence in the recommendation
    """
    recommendation_id: str
    action_type: str
    target: str
    current_value: float
    proposed_value: float
    expected_benefit: float
    uncertainty: float
    risk_level: RiskLevel
    confidence: float


@dataclass
class UncertaintyThresholds:
    """
    Configurable thresholds for uncertainty gating.

    Attributes:
        passed_threshold: Below this, uncertainty is acceptable
        warning_threshold: Above passed, below this, generate warning
        blocked_threshold: Above this, block recommendation
        confirmation_threshold: Above this, require operator confirmation
        risk_multipliers: Multipliers for different risk levels
    """
    passed_threshold: float = 5.0   # 5% uncertainty
    warning_threshold: float = 10.0  # 10% uncertainty
    blocked_threshold: float = 20.0  # 20% uncertainty
    confirmation_threshold: float = 15.0  # 15% for high-risk

    # Risk-based multipliers (stricter for higher risk)
    risk_multipliers: Dict[RiskLevel, float] = field(default_factory=lambda: {
        RiskLevel.LOW: 1.5,       # More lenient
        RiskLevel.MEDIUM: 1.0,   # Baseline
        RiskLevel.HIGH: 0.7,     # Stricter
        RiskLevel.CRITICAL: 0.5  # Very strict
    })

    def get_effective_threshold(
        self,
        base_threshold: float,
        risk_level: RiskLevel
    ) -> float:
        """Get threshold adjusted for risk level."""
        multiplier = self.risk_multipliers.get(risk_level, 1.0)
        return base_threshold * multiplier


class UncertaintyGate:
    """
    Quality gate for uncertainty-based decision making.

    Gates recommendations based on uncertainty levels, requiring
    operator confirmation for high-uncertainty scenarios and
    blocking recommendations when uncertainty is unacceptably high.

    Example:
        gate = UncertaintyGate()

        # Check if recommendation can proceed
        result = gate.check_recommendation_confidence(
            recommendation=my_recommendation,
            min_confidence=0.90
        )

        if result.is_blocked():
            print(f"Blocked: {result.reason}")
        elif result.requires_human_action():
            print(f"Requires confirmation: {result.required_action}")
        else:
            # Proceed with recommendation
            pass
    """

    def __init__(self, thresholds: Optional[UncertaintyThresholds] = None):
        """
        Initialize uncertainty gate.

        Args:
            thresholds: Custom thresholds (default: standard thresholds)
        """
        self.thresholds = thresholds or UncertaintyThresholds()
        self._gate_history: List[GateResult] = []
        self._suppressed_warnings: Dict[str, datetime] = {}

    def check_recommendation_confidence(
        self,
        recommendation: Recommendation,
        min_confidence: float = 0.90
    ) -> GateResult:
        """
        Check if recommendation meets minimum confidence requirements.

        Args:
            recommendation: Recommendation to check
            min_confidence: Minimum required confidence (0-1)

        Returns:
            GateResult indicating whether recommendation can proceed
        """
        # Get effective thresholds based on risk level
        effective_warning = self.thresholds.get_effective_threshold(
            self.thresholds.warning_threshold,
            recommendation.risk_level
        )
        effective_blocked = self.thresholds.get_effective_threshold(
            self.thresholds.blocked_threshold,
            recommendation.risk_level
        )
        effective_confirmation = self.thresholds.get_effective_threshold(
            self.thresholds.confirmation_threshold,
            recommendation.risk_level
        )

        # Calculate relative uncertainty
        if abs(recommendation.expected_benefit) < 1e-10:
            relative_uncertainty = 100.0  # Very high for near-zero benefit
        else:
            relative_uncertainty = (
                recommendation.uncertainty / abs(recommendation.expected_benefit)
            ) * 100.0

        # Determine gate status
        if recommendation.confidence < min_confidence:
            # Confidence below minimum
            if relative_uncertainty >= effective_blocked:
                status = GateStatus.BLOCKED
                reason = (
                    f"Recommendation blocked: confidence {recommendation.confidence:.1%} "
                    f"below minimum {min_confidence:.1%} and uncertainty "
                    f"{relative_uncertainty:.1f}% exceeds limit {effective_blocked:.1f}%"
                )
                required_action = "Reduce measurement uncertainty or increase data quality"

            elif relative_uncertainty >= effective_confirmation:
                status = GateStatus.REQUIRES_CONFIRMATION
                reason = (
                    f"Operator confirmation required: confidence {recommendation.confidence:.1%} "
                    f"below minimum {min_confidence:.1%}"
                )
                required_action = (
                    "Review recommendation details and confirm to proceed"
                )

            else:
                status = GateStatus.WARNING
                reason = (
                    f"Recommendation has reduced confidence: {recommendation.confidence:.1%} "
                    f"(minimum: {min_confidence:.1%})"
                )
                required_action = ""

        elif relative_uncertainty >= effective_blocked:
            # High uncertainty blocks even with good confidence
            status = GateStatus.BLOCKED
            reason = (
                f"Recommendation blocked: uncertainty {relative_uncertainty:.1f}% "
                f"exceeds limit {effective_blocked:.1f}% for "
                f"{recommendation.risk_level.value} risk action"
            )
            required_action = "Improve measurement quality before proceeding"

        elif relative_uncertainty >= effective_confirmation:
            # Medium-high uncertainty requires confirmation for high-risk
            if recommendation.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                status = GateStatus.REQUIRES_CONFIRMATION
                reason = (
                    f"Operator confirmation required for {recommendation.risk_level.value} "
                    f"risk action with {relative_uncertainty:.1f}% uncertainty"
                )
                required_action = "Review and confirm to proceed"
            else:
                status = GateStatus.WARNING
                reason = (
                    f"Elevated uncertainty ({relative_uncertainty:.1f}%) for recommendation"
                )
                required_action = ""

        elif relative_uncertainty >= effective_warning:
            # Warning level
            status = GateStatus.WARNING
            reason = (
                f"Uncertainty ({relative_uncertainty:.1f}%) approaching threshold"
            )
            required_action = ""

        else:
            # All checks passed
            status = GateStatus.PASSED
            reason = (
                f"Recommendation meets quality requirements: confidence "
                f"{recommendation.confidence:.1%}, uncertainty {relative_uncertainty:.1f}%"
            )
            required_action = ""

        result = GateResult(
            status=status,
            recommendation_id=recommendation.recommendation_id,
            uncertainty_level=relative_uncertainty,
            threshold=effective_blocked,
            confidence_level=recommendation.confidence,
            reason=reason,
            required_action=required_action
        )

        # Log gate result
        self._gate_history.append(result)

        logger.info(
            f"Gate check for {recommendation.recommendation_id}: "
            f"{status.value} (uncertainty: {relative_uncertainty:.1f}%, "
            f"confidence: {recommendation.confidence:.1%})"
        )

        return result

    def check_measurement_quality(
        self,
        measurements: Dict[str, UncertainValue],
        thresholds: Optional[Dict[str, float]] = None
    ) -> QualityCheckResult:
        """
        Check quality of a set of measurements.

        Args:
            measurements: Dictionary of measurement name to UncertainValue
            thresholds: Custom per-measurement thresholds (optional)

        Returns:
            QualityCheckResult with overall and per-measurement status
        """
        thresholds = thresholds or {}
        default_threshold = self.thresholds.warning_threshold

        details = {}
        passed_count = 0
        warning_count = 0
        failed_count = 0
        worst_uncertainty = 0.0
        worst_measurement = ""
        recommendations = []

        for name, value in measurements.items():
            threshold = thresholds.get(name, default_threshold)
            relative_uncertainty = value.relative_uncertainty()

            if relative_uncertainty >= self.thresholds.blocked_threshold:
                status = "failed"
                failed_count += 1
                recommendations.append(
                    f"Recalibrate or replace sensor for {name} "
                    f"(uncertainty: {relative_uncertainty:.1f}%)"
                )
            elif relative_uncertainty >= threshold:
                status = "warning"
                warning_count += 1
                recommendations.append(
                    f"Monitor sensor for {name} "
                    f"(uncertainty approaching limit: {relative_uncertainty:.1f}%)"
                )
            else:
                status = "passed"
                passed_count += 1

            details[name] = {
                "status": status,
                "uncertainty_percent": relative_uncertainty,
                "threshold": threshold,
                "value": value.mean,
                "std": value.std
            }

            if relative_uncertainty > worst_uncertainty:
                worst_uncertainty = relative_uncertainty
                worst_measurement = name

        # Determine overall status
        if failed_count > 0:
            overall_status = GateStatus.BLOCKED
        elif warning_count > 0:
            overall_status = GateStatus.WARNING
        else:
            overall_status = GateStatus.PASSED

        return QualityCheckResult(
            overall_status=overall_status,
            checked_measurements=len(measurements),
            passed_count=passed_count,
            warning_count=warning_count,
            failed_count=failed_count,
            details=details,
            worst_measurement=worst_measurement,
            recommendations=recommendations
        )

    def require_operator_confirmation(
        self,
        uncertainty_level: float,
        risk_level: RiskLevel = RiskLevel.MEDIUM
    ) -> bool:
        """
        Determine if operator confirmation is required.

        Args:
            uncertainty_level: Current uncertainty level (%)
            risk_level: Risk level of the action

        Returns:
            True if operator confirmation is required
        """
        effective_threshold = self.thresholds.get_effective_threshold(
            self.thresholds.confirmation_threshold,
            risk_level
        )

        # Always require confirmation for critical actions with any uncertainty
        if risk_level == RiskLevel.CRITICAL and uncertainty_level > 5.0:
            return True

        # Require confirmation above threshold
        return uncertainty_level >= effective_threshold

    def generate_uncertainty_warning(
        self,
        high_uncertainty_sources: List[Tuple[str, float, str]]
    ) -> Warning:
        """
        Generate a warning for high uncertainty sources.

        Args:
            high_uncertainty_sources: List of (source_id, uncertainty, source_type)

        Returns:
            Warning object for display
        """
        if not high_uncertainty_sources:
            # No high uncertainty sources
            return Warning(
                warning_id=f"warn_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                priority=WarningPriority.INFO,
                title="Uncertainty Within Limits",
                message="All uncertainty sources are within acceptable limits.",
                affected_sensors=[],
                affected_calculations=[],
                recommended_actions=[]
            )

        # Sort by uncertainty level
        sorted_sources = sorted(
            high_uncertainty_sources,
            key=lambda x: x[1],
            reverse=True
        )

        # Determine priority based on worst uncertainty
        worst_uncertainty = sorted_sources[0][1]
        if worst_uncertainty >= self.thresholds.blocked_threshold:
            priority = WarningPriority.CRITICAL
        elif worst_uncertainty >= self.thresholds.confirmation_threshold:
            priority = WarningPriority.HIGH
        elif worst_uncertainty >= self.thresholds.warning_threshold:
            priority = WarningPriority.MEDIUM
        else:
            priority = WarningPriority.LOW

        # Build affected lists
        affected_sensors = []
        affected_calculations = []

        for source_id, uncertainty, source_type in sorted_sources:
            if source_type == "sensor":
                affected_sensors.append(source_id)
            else:
                affected_calculations.append(source_id)

        # Build recommendations
        recommended_actions = []

        for source_id, uncertainty, source_type in sorted_sources[:3]:  # Top 3
            if source_type == "sensor":
                if uncertainty >= self.thresholds.blocked_threshold:
                    recommended_actions.append(
                        f"URGENT: Recalibrate sensor {source_id} "
                        f"(uncertainty: {uncertainty:.1f}%)"
                    )
                else:
                    recommended_actions.append(
                        f"Schedule calibration for sensor {source_id} "
                        f"(uncertainty: {uncertainty:.1f}%)"
                    )
            else:
                recommended_actions.append(
                    f"Review input data quality for {source_id}"
                )

        # Build message
        source_summary = ", ".join(
            f"{s[0]} ({s[1]:.1f}%)"
            for s in sorted_sources[:5]
        )

        warning = Warning(
            warning_id=f"warn_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            priority=priority,
            title=f"High Uncertainty Detected ({len(sorted_sources)} sources)",
            message=(
                f"The following sources have elevated uncertainty: {source_summary}. "
                f"This may affect the reliability of calculations and recommendations."
            ),
            affected_sensors=affected_sensors,
            affected_calculations=affected_calculations,
            recommended_actions=recommended_actions
        )

        return warning

    def check_propagated_uncertainty(
        self,
        propagated: PropagatedUncertainty,
        acceptable_uncertainty: float = 10.0
    ) -> GateResult:
        """
        Check if propagated uncertainty is acceptable.

        Args:
            propagated: Propagated uncertainty result
            acceptable_uncertainty: Maximum acceptable uncertainty (%)

        Returns:
            GateResult for the propagated value
        """
        relative_uncertainty = propagated.relative_uncertainty_percent()

        if relative_uncertainty <= self.thresholds.passed_threshold:
            status = GateStatus.PASSED
            reason = f"Propagated uncertainty ({relative_uncertainty:.1f}%) is acceptable"
            required_action = ""

        elif relative_uncertainty <= acceptable_uncertainty:
            status = GateStatus.WARNING
            reason = (
                f"Propagated uncertainty ({relative_uncertainty:.1f}%) is elevated "
                f"but within acceptable limit ({acceptable_uncertainty:.1f}%)"
            )
            required_action = ""

        elif relative_uncertainty <= self.thresholds.blocked_threshold:
            status = GateStatus.REQUIRES_CONFIRMATION
            reason = (
                f"Propagated uncertainty ({relative_uncertainty:.1f}%) exceeds "
                f"acceptable limit ({acceptable_uncertainty:.1f}%)"
            )
            required_action = (
                f"Review dominant contributor ({propagated.dominant_contributor}) "
                f"for potential improvement"
            )

        else:
            status = GateStatus.BLOCKED
            reason = (
                f"Propagated uncertainty ({relative_uncertainty:.1f}%) is too high "
                f"for reliable decision making"
            )
            required_action = (
                f"Reduce uncertainty in {propagated.dominant_contributor} "
                f"before proceeding"
            )

        return GateResult(
            status=status,
            recommendation_id=propagated.output_name,
            uncertainty_level=relative_uncertainty,
            threshold=acceptable_uncertainty,
            confidence_level=1.0 - (relative_uncertainty / 100.0),
            reason=reason,
            required_action=required_action
        )

    def get_gate_history(
        self,
        since: Optional[datetime] = None,
        status_filter: Optional[GateStatus] = None
    ) -> List[GateResult]:
        """
        Get history of gate decisions.

        Args:
            since: Only return results after this time
            status_filter: Only return results with this status

        Returns:
            List of GateResult objects
        """
        results = self._gate_history.copy()

        if since:
            results = [r for r in results if r.timestamp >= since]

        if status_filter:
            results = [r for r in results if r.status == status_filter]

        return results

    def suppress_warning(self, warning_id: str, duration_hours: int = 24) -> None:
        """
        Suppress a warning for specified duration.

        Args:
            warning_id: Warning to suppress
            duration_hours: Hours to suppress
        """
        from datetime import timedelta
        suppress_until = datetime.utcnow() + timedelta(hours=duration_hours)
        self._suppressed_warnings[warning_id] = suppress_until

        logger.info(f"Suppressed warning {warning_id} until {suppress_until}")

    def is_warning_suppressed(self, warning_id: str) -> bool:
        """Check if warning is currently suppressed."""
        if warning_id not in self._suppressed_warnings:
            return False

        suppress_until = self._suppressed_warnings[warning_id]
        if datetime.utcnow() >= suppress_until:
            # Suppression expired
            del self._suppressed_warnings[warning_id]
            return False

        return True

    def clear_gate_history(self) -> None:
        """Clear gate decision history."""
        self._gate_history.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about gate decisions.

        Returns:
            Dictionary with gate statistics
        """
        total = len(self._gate_history)
        if total == 0:
            return {
                "total_checks": 0,
                "passed": 0,
                "warnings": 0,
                "blocked": 0,
                "confirmations_required": 0
            }

        passed = sum(1 for r in self._gate_history if r.status == GateStatus.PASSED)
        warnings = sum(1 for r in self._gate_history if r.status == GateStatus.WARNING)
        blocked = sum(1 for r in self._gate_history if r.status == GateStatus.BLOCKED)
        confirmations = sum(
            1 for r in self._gate_history
            if r.status == GateStatus.REQUIRES_CONFIRMATION
        )

        return {
            "total_checks": total,
            "passed": passed,
            "passed_percent": (passed / total) * 100,
            "warnings": warnings,
            "warnings_percent": (warnings / total) * 100,
            "blocked": blocked,
            "blocked_percent": (blocked / total) * 100,
            "confirmations_required": confirmations,
            "confirmations_percent": (confirmations / total) * 100
        }
