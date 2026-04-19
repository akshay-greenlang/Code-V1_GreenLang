"""
GL-004 Burnmaster - Recommendation Engine

Generates, prioritizes, and tracks optimization recommendations.

Features:
    - generate_recommendations: Create actionable recommendations
    - prioritize_recommendations: Sort by impact and safety
    - explain_recommendation: Generate human-readable explanations
    - track_recommendation_acceptance: Learn from operator feedback

Author: GreenLang Optimization Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import uuid

from pydantic import BaseModel, Field

from .combustion_optimizer import OptimizationResult
from .setpoint_optimizer import SetpointRecommendation

logger = logging.getLogger(__name__)


class RecommendationType(str, Enum):
    """Type of recommendation."""
    SETPOINT_CHANGE = "setpoint_change"
    LOAD_CHANGE = "load_change"
    MAINTENANCE = "maintenance"
    ALERT = "alert"
    OPTIMIZATION = "optimization"
    SAFETY = "safety"


class RecommendationStatus(str, Enum):
    """Status of recommendation."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"
    IMPLEMENTED = "implemented"
    FAILED = "failed"


class ImpactLevel(str, Enum):
    """Impact level of recommendation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Recommendation(BaseModel):
    """Actionable recommendation for operators."""
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Classification
    recommendation_type: RecommendationType = Field(default=RecommendationType.OPTIMIZATION)
    status: RecommendationStatus = Field(default=RecommendationStatus.PENDING)
    impact_level: ImpactLevel = Field(default=ImpactLevel.MEDIUM)

    # Content
    title: str = Field(..., description="Short title")
    description: str = Field(..., description="Detailed description")
    action_items: List[str] = Field(default_factory=list, description="Steps to implement")

    # Setpoint changes
    setpoint_changes: Dict[str, float] = Field(default_factory=dict, description="Variable: new_value")
    current_values: Dict[str, float] = Field(default_factory=dict, description="Variable: current_value")

    # Expected benefits
    expected_savings_per_hour: float = Field(default=0.0)
    expected_efficiency_improvement: float = Field(default=0.0, description="Percentage")
    expected_emissions_reduction: float = Field(default=0.0, description="Percentage")

    # Uncertainty
    confidence: float = Field(default=0.95, ge=0, le=1)
    uncertainty_range: Tuple[float, float] = Field(default=(0.0, 0.0))

    # Constraints
    constraint_margins: Dict[str, float] = Field(default_factory=dict)
    safety_verified: bool = Field(default=False)

    # Timing
    valid_until: Optional[datetime] = Field(default=None)
    implementation_time_s: float = Field(default=60.0, ge=0)

    # Tracking
    created_by: str = Field(default="optimizer")
    priority_score: float = Field(default=0.0)

    # Provenance
    source_optimization_id: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        if not self.provenance_hash:
            changes_str = ",".join([f"{k}:{v:.4f}" for k, v in self.setpoint_changes.items()])
            hash_input = f"{self.recommendation_id}|{self.title}|{changes_str}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    @property
    def is_expired(self) -> bool:
        """Check if recommendation has expired."""
        if self.valid_until is None:
            return False
        return datetime.now(timezone.utc) > self.valid_until


class ExplanationPayload(BaseModel):
    """Human-readable explanation of a recommendation."""
    explanation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    recommendation_id: str = Field(...)

    # Summary
    summary: str = Field(..., description="One-sentence summary")
    detailed_explanation: str = Field(..., description="Full explanation")

    # Why this change
    problem_identified: str = Field(default="", description="What problem was identified")
    root_cause: str = Field(default="", description="Root cause analysis")

    # What to do
    recommended_actions: List[str] = Field(default_factory=list)
    expected_outcome: str = Field(default="")

    # Supporting data
    supporting_metrics: Dict[str, float] = Field(default_factory=dict)
    trend_analysis: str = Field(default="")

    # Risk assessment
    risk_if_not_implemented: str = Field(default="")
    risk_of_implementation: str = Field(default="")

    # References
    related_standards: List[str] = Field(default_factory=list)


class RecommendationOutcome(BaseModel):
    """Outcome of an implemented recommendation."""
    outcome_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    recommendation_id: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Status
    was_accepted: bool = Field(...)
    was_implemented: bool = Field(default=False)
    implementation_time: Optional[datetime] = Field(default=None)

    # Actual results
    actual_savings_per_hour: Optional[float] = Field(default=None)
    actual_efficiency_change: Optional[float] = Field(default=None)
    actual_emissions_change: Optional[float] = Field(default=None)

    # Comparison
    savings_vs_expected: Optional[float] = Field(default=None)
    efficiency_vs_expected: Optional[float] = Field(default=None)

    # Feedback
    operator_feedback: str = Field(default="")
    rejection_reason: Optional[str] = Field(default=None)

    # Learning
    was_prediction_accurate: bool = Field(default=True)
    error_margin: float = Field(default=0.0)


class RecommendationEngine:
    """
    Engine for generating and managing optimization recommendations.

    Features:
    - Converts optimization results to actionable recommendations
    - Prioritizes based on impact, safety, and confidence
    - Provides human-readable explanations
    - Tracks outcomes for continuous learning
    """

    def __init__(
        self,
        min_savings_threshold: float = 1.0,
        min_efficiency_threshold: float = 0.1,
        recommendation_validity_hours: float = 1.0
    ) -> None:
        """
        Initialize recommendation engine.

        Args:
            min_savings_threshold: Minimum savings ($/hr) to generate recommendation
            min_efficiency_threshold: Minimum efficiency improvement (%)
            recommendation_validity_hours: How long recommendations are valid
        """
        self.min_savings_threshold = min_savings_threshold
        self.min_efficiency_threshold = min_efficiency_threshold
        self.recommendation_validity_hours = recommendation_validity_hours

        # Storage for tracking
        self._recommendations: Dict[str, Recommendation] = {}
        self._outcomes: Dict[str, RecommendationOutcome] = {}
        self._acceptance_history: List[Tuple[str, bool]] = []

        # Learning statistics
        self._prediction_accuracy: List[float] = []

        logger.info("RecommendationEngine initialized")

    def generate_recommendations(
        self,
        optimization_result: OptimizationResult
    ) -> List[Recommendation]:
        """
        Generate recommendations from optimization result.

        Args:
            optimization_result: Result from CombustionOptimizer

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        if not optimization_result.optimal_setpoints:
            return recommendations

        if not optimization_result.is_feasible:
            # Generate safety recommendation
            rec = self._create_safety_recommendation(optimization_result)
            if rec:
                recommendations.append(rec)
            return recommendations

        # Check if improvement is significant
        if optimization_result.improvement_percent < self.min_efficiency_threshold:
            return recommendations

        # Create setpoint change recommendation
        optimal = optimization_result.optimal_setpoints
        current = optimization_result.current_setpoints

        setpoint_changes = {}
        current_values = {}

        if current:
            if abs(optimal.o2_setpoint_percent - current.o2_setpoint_percent) > 0.1:
                setpoint_changes["o2_setpoint_percent"] = optimal.o2_setpoint_percent
                current_values["o2_setpoint_percent"] = current.o2_setpoint_percent

            if abs(optimal.air_damper_position - current.air_damper_position) > 1.0:
                setpoint_changes["air_damper_position"] = optimal.air_damper_position
                current_values["air_damper_position"] = current.air_damper_position

            if abs(optimal.fuel_valve_position - current.fuel_valve_position) > 1.0:
                setpoint_changes["fuel_valve_position"] = optimal.fuel_valve_position
                current_values["fuel_valve_position"] = current.fuel_valve_position

        if not setpoint_changes:
            return recommendations

        # Determine impact level
        if optimization_result.improvement_percent > 5:
            impact_level = ImpactLevel.HIGH
        elif optimization_result.improvement_percent > 2:
            impact_level = ImpactLevel.MEDIUM
        else:
            impact_level = ImpactLevel.LOW

        # Create action items
        action_items = []
        for var, new_val in setpoint_changes.items():
            old_val = current_values.get(var, 0)
            direction = "increase" if new_val > old_val else "decrease"
            action_items.append(
                f"{direction.capitalize()} {var.replace('_', ' ')} from {old_val:.1f} to {new_val:.1f}"
            )

        # Create recommendation
        validity = datetime.now(timezone.utc) + \
            __import__("datetime").timedelta(hours=self.recommendation_validity_hours)

        rec = Recommendation(
            recommendation_type=RecommendationType.OPTIMIZATION,
            impact_level=impact_level,
            title="Combustion Optimization Opportunity",
            description=f"Optimization can improve efficiency by {optimization_result.improvement_percent:.1f}%",
            action_items=action_items,
            setpoint_changes=setpoint_changes,
            current_values=current_values,
            expected_savings_per_hour=optimization_result.savings_per_hour,
            expected_efficiency_improvement=optimization_result.improvement_percent,
            confidence=0.95,
            uncertainty_range=optimization_result.objective_result.total_cost_uncertainty
            if optimization_result.objective_result else (0.0, 0.0),
            safety_verified=True,
            valid_until=validity,
            implementation_time_s=60.0,
            source_optimization_id=optimization_result.result_id
        )

        # Calculate priority score
        rec.priority_score = self._calculate_priority_score(rec)

        recommendations.append(rec)
        self._recommendations[rec.recommendation_id] = rec

        logger.info("Generated recommendation: %s (priority=%.2f)",
                    rec.title, rec.priority_score)

        return recommendations

    def _create_safety_recommendation(
        self,
        optimization_result: OptimizationResult
    ) -> Optional[Recommendation]:
        """Create recommendation for safety-related issues."""
        if not optimization_result.constraint_result:
            return None

        constraint_result = optimization_result.constraint_result

        if constraint_result.is_feasible:
            return None

        # Find the most violated constraint
        max_violation_name = constraint_result.max_violation_constraint

        return Recommendation(
            recommendation_type=RecommendationType.SAFETY,
            impact_level=ImpactLevel.CRITICAL,
            title="Safety Constraint Violation Detected",
            description=f"Constraint '{max_violation_name}' is violated. Immediate attention required.",
            action_items=[
                f"Review {max_violation_name} constraint",
                "Adjust setpoints to restore safe operation",
                "Contact operations supervisor if violation persists"
            ],
            safety_verified=False,
            valid_until=datetime.now(timezone.utc) + __import__("datetime").timedelta(minutes=15),
            priority_score=100.0  # Maximum priority for safety
        )

    def prioritize_recommendations(
        self,
        recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        """
        Prioritize recommendations by importance.

        Args:
            recommendations: List of recommendations to prioritize

        Returns:
            Sorted list (highest priority first)
        """
        for rec in recommendations:
            rec.priority_score = self._calculate_priority_score(rec)

        return sorted(recommendations, key=lambda r: r.priority_score, reverse=True)

    def _calculate_priority_score(self, rec: Recommendation) -> float:
        """Calculate priority score for a recommendation."""
        score = 0.0

        # Impact level contribution (0-40 points)
        impact_scores = {
            ImpactLevel.CRITICAL: 40,
            ImpactLevel.HIGH: 30,
            ImpactLevel.MEDIUM: 20,
            ImpactLevel.LOW: 10
        }
        score += impact_scores.get(rec.impact_level, 10)

        # Safety recommendations get bonus (0-30 points)
        if rec.recommendation_type == RecommendationType.SAFETY:
            score += 30

        # Savings contribution (0-20 points)
        if rec.expected_savings_per_hour > 0:
            savings_score = min(20, rec.expected_savings_per_hour * 2)
            score += savings_score

        # Confidence contribution (0-10 points)
        score += rec.confidence * 10

        # Constraint margin penalty (-10 to 0)
        if rec.constraint_margins:
            min_margin = min(rec.constraint_margins.values())
            if min_margin < 0.1:
                score -= 10 * (0.1 - min_margin)

        return score

    def explain_recommendation(
        self,
        rec: Recommendation
    ) -> ExplanationPayload:
        """
        Generate human-readable explanation.

        Args:
            rec: Recommendation to explain

        Returns:
            ExplanationPayload with detailed explanation
        """
        # Build summary
        summary = f"{rec.title}: {rec.description}"

        # Build detailed explanation
        detailed_parts = []

        if rec.recommendation_type == RecommendationType.SAFETY:
            detailed_parts.append(
                "SAFETY ALERT: This recommendation addresses a constraint violation "
                "that requires immediate attention."
            )
        else:
            detailed_parts.append(
                f"This optimization recommendation is expected to save "
                f"${rec.expected_savings_per_hour:.2f}/hr by improving combustion efficiency "
                f"by {rec.expected_efficiency_improvement:.1f}%."
            )

        if rec.setpoint_changes:
            detailed_parts.append("\nRecommended setpoint changes:")
            for var, new_val in rec.setpoint_changes.items():
                old_val = rec.current_values.get(var, 0)
                change = new_val - old_val
                detailed_parts.append(
                    f"  - {var.replace('_', ' ')}: {old_val:.1f} -> {new_val:.1f} "
                    f"({'+' if change > 0 else ''}{change:.1f})"
                )

        detailed_explanation = "\n".join(detailed_parts)

        # Problem and cause analysis
        problem = self._identify_problem(rec)
        root_cause = self._analyze_root_cause(rec)

        # Expected outcome
        expected_outcome = (
            f"Implementing these changes is expected to improve efficiency by "
            f"{rec.expected_efficiency_improvement:.1f}% and save "
            f"${rec.expected_savings_per_hour:.2f} per hour."
        )

        # Risks
        risk_not_implemented = (
            "Continuing with current setpoints may result in suboptimal efficiency "
            "and higher operating costs."
        )
        risk_implementation = (
            "Setpoint changes should be implemented gradually to avoid process upsets. "
            "Monitor CO and flame stability during implementation."
        )

        return ExplanationPayload(
            recommendation_id=rec.recommendation_id,
            summary=summary,
            detailed_explanation=detailed_explanation,
            problem_identified=problem,
            root_cause=root_cause,
            recommended_actions=rec.action_items,
            expected_outcome=expected_outcome,
            supporting_metrics={
                "efficiency_improvement": rec.expected_efficiency_improvement,
                "savings_per_hour": rec.expected_savings_per_hour,
                "confidence": rec.confidence
            },
            risk_if_not_implemented=risk_not_implemented,
            risk_of_implementation=risk_implementation,
            related_standards=["EPA 40 CFR Part 60", "NFPA 85"]
        )

    def _identify_problem(self, rec: Recommendation) -> str:
        """Identify the problem being addressed."""
        if rec.recommendation_type == RecommendationType.SAFETY:
            return "A safety constraint is being violated."

        changes = rec.setpoint_changes
        if "o2_setpoint_percent" in changes:
            return "O2 levels are not optimal for current operating conditions."
        if "air_damper_position" in changes:
            return "Air flow is not properly matched to fuel flow."

        return "Combustion setpoints are not optimized."

    def _analyze_root_cause(self, rec: Recommendation) -> str:
        """Analyze root cause of the issue."""
        changes = rec.setpoint_changes
        current = rec.current_values

        if "o2_setpoint_percent" in changes:
            current_o2 = current.get("o2_setpoint_percent", 0)
            new_o2 = changes["o2_setpoint_percent"]
            if new_o2 > current_o2:
                return "Excess air is too low, potentially causing incomplete combustion."
            else:
                return "Excess air is too high, reducing thermal efficiency."

        return "Operating conditions have changed, requiring setpoint adjustments."

    def track_recommendation_acceptance(
        self,
        rec_id: str,
        accepted: bool,
        outcome: Dict[str, Any]
    ) -> None:
        """
        Track recommendation acceptance and outcomes.

        Args:
            rec_id: Recommendation ID
            accepted: Whether recommendation was accepted
            outcome: Dictionary with outcome data
        """
        rec = self._recommendations.get(rec_id)
        if not rec:
            logger.warning("Recommendation not found: %s", rec_id)
            return

        # Update status
        rec.status = RecommendationStatus.ACCEPTED if accepted else RecommendationStatus.REJECTED

        # Create outcome record
        rec_outcome = RecommendationOutcome(
            recommendation_id=rec_id,
            was_accepted=accepted,
            was_implemented=outcome.get("implemented", False),
            actual_savings_per_hour=outcome.get("actual_savings"),
            actual_efficiency_change=outcome.get("actual_efficiency"),
            actual_emissions_change=outcome.get("actual_emissions"),
            operator_feedback=outcome.get("feedback", ""),
            rejection_reason=outcome.get("rejection_reason")
        )

        # Calculate prediction accuracy if implemented
        if rec_outcome.was_implemented and rec_outcome.actual_savings_per_hour is not None:
            if rec.expected_savings_per_hour > 0:
                accuracy = 1.0 - abs(
                    rec_outcome.actual_savings_per_hour - rec.expected_savings_per_hour
                ) / rec.expected_savings_per_hour
                rec_outcome.was_prediction_accurate = accuracy > 0.8
                rec_outcome.error_margin = 1.0 - accuracy
                self._prediction_accuracy.append(accuracy)

            rec_outcome.savings_vs_expected = (
                rec_outcome.actual_savings_per_hour - rec.expected_savings_per_hour
            )

        self._outcomes[rec_id] = rec_outcome
        self._acceptance_history.append((rec_id, accepted))

        logger.info(
            "Tracked recommendation %s: accepted=%s, implemented=%s",
            rec_id, accepted, rec_outcome.was_implemented
        )

    def get_recommendation(self, rec_id: str) -> Optional[Recommendation]:
        """Get recommendation by ID."""
        return self._recommendations.get(rec_id)

    def get_pending_recommendations(self) -> List[Recommendation]:
        """Get all pending recommendations."""
        return [
            r for r in self._recommendations.values()
            if r.status == RecommendationStatus.PENDING and not r.is_expired
        ]

    def get_acceptance_rate(self) -> float:
        """Get overall recommendation acceptance rate."""
        if not self._acceptance_history:
            return 0.0
        accepted = sum(1 for _, a in self._acceptance_history if a)
        return accepted / len(self._acceptance_history)

    def get_prediction_accuracy(self) -> float:
        """Get average prediction accuracy."""
        if not self._prediction_accuracy:
            return 0.0
        return sum(self._prediction_accuracy) / len(self._prediction_accuracy)
