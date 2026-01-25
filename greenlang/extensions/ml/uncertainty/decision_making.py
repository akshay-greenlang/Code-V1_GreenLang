# -*- coding: utf-8 -*-
"""
Uncertainty-Aware Decision Making Module

This module provides decision-making capabilities that incorporate
uncertainty quantification for GreenLang ML models, enabling confidence
thresholds, risk-adjusted recommendations, human-in-the-loop triggers,
and multi-objective optimization under uncertainty.

Uncertainty-aware decision making is critical for regulatory compliance
where automated actions must be justified with appropriate confidence
levels and audit trails.

Example:
    >>> from greenlang.ml.uncertainty.decision_making import DecisionEngine
    >>> engine = DecisionEngine(config=DecisionConfig(
    ...     confidence_threshold=0.90,
    ...     risk_tolerance="conservative"
    ... ))
    >>> decision = engine.make_decision(prediction, uncertainty)
    >>> if decision.requires_human_review:
    ...     escalate_to_human(decision)
"""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pydantic import BaseModel, Field
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import uuid

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Configuration
# ============================================================================

class RiskTolerance(str, Enum):
    """Risk tolerance levels."""
    AGGRESSIVE = "aggressive"
    MODERATE = "moderate"
    CONSERVATIVE = "conservative"
    VERY_CONSERVATIVE = "very_conservative"


class ActionType(str, Enum):
    """Types of automated actions."""
    APPROVE = "approve"
    REJECT = "reject"
    DEFER = "defer"
    HUMAN_REVIEW = "human_review"
    ESCALATE = "escalate"
    MONITOR = "monitor"


class DecisionOutcome(str, Enum):
    """Decision outcomes."""
    AUTOMATED = "automated"
    HUMAN_REQUIRED = "human_required"
    PENDING = "pending"
    UNCERTAIN = "uncertain"


class ObjectiveType(str, Enum):
    """Types of objectives for optimization."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"


class DecisionConfig(BaseModel):
    """Configuration for decision engine."""

    confidence_threshold: float = Field(
        default=0.90,
        ge=0.5,
        le=0.99,
        description="Minimum confidence for automated action"
    )
    risk_tolerance: RiskTolerance = Field(
        default=RiskTolerance.CONSERVATIVE,
        description="Risk tolerance level"
    )
    uncertainty_threshold: float = Field(
        default=0.2,
        gt=0,
        description="Maximum uncertainty for automated action"
    )
    human_review_threshold: float = Field(
        default=0.7,
        ge=0.5,
        le=0.95,
        description="Confidence below which human review is required"
    )
    escalation_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Confidence below which escalation is required"
    )
    cost_of_false_positive: float = Field(
        default=1.0,
        gt=0,
        description="Relative cost of false positive"
    )
    cost_of_false_negative: float = Field(
        default=1.0,
        gt=0,
        description="Relative cost of false negative"
    )
    enable_audit_trail: bool = Field(
        default=True,
        description="Enable decision audit trail"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )


# ============================================================================
# Decision Models
# ============================================================================

class ConfidenceLevel(BaseModel):
    """Confidence level information."""

    score: float = Field(
        ...,
        description="Confidence score (0-1)"
    )
    level: str = Field(
        ...,
        description="Categorical level (very_high, high, medium, low, very_low)"
    )
    threshold_met: bool = Field(
        ...,
        description="Whether threshold is met for automation"
    )


class RiskAssessment(BaseModel):
    """Risk assessment for a decision."""

    risk_score: float = Field(
        ...,
        description="Overall risk score (0-1)"
    )
    risk_level: str = Field(
        ...,
        description="Risk level (low, medium, high, critical)"
    )
    risk_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual risk factors"
    )
    expected_loss: float = Field(
        ...,
        description="Expected loss given uncertainty"
    )
    worst_case_loss: float = Field(
        ...,
        description="Worst case loss"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Risk mitigation recommendations"
    )


class Decision(BaseModel):
    """A decision made by the engine."""

    decision_id: str = Field(
        ...,
        description="Unique decision identifier"
    )
    prediction: float = Field(
        ...,
        description="Original prediction"
    )
    uncertainty: float = Field(
        ...,
        description="Prediction uncertainty"
    )
    confidence: ConfidenceLevel = Field(
        ...,
        description="Confidence information"
    )
    action: ActionType = Field(
        ...,
        description="Recommended action"
    )
    outcome: DecisionOutcome = Field(
        ...,
        description="Decision outcome"
    )
    risk_assessment: RiskAssessment = Field(
        ...,
        description="Risk assessment"
    )
    requires_human_review: bool = Field(
        ...,
        description="Whether human review is required"
    )
    justification: str = Field(
        ...,
        description="Justification for decision"
    )
    alternatives: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative actions considered"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 provenance hash"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Decision timestamp"
    )


class HumanReviewRequest(BaseModel):
    """Request for human review."""

    request_id: str = Field(
        ...,
        description="Request identifier"
    )
    decision: Decision = Field(
        ...,
        description="Original decision"
    )
    reason: str = Field(
        ...,
        description="Reason for human review"
    )
    priority: str = Field(
        default="normal",
        description="Review priority (low, normal, high, urgent)"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )
    suggested_action: Optional[ActionType] = Field(
        default=None,
        description="Suggested action from model"
    )
    confidence_details: str = Field(
        default="",
        description="Detailed confidence explanation"
    )
    deadline: Optional[datetime] = Field(
        default=None,
        description="Review deadline"
    )


class DecisionAuditEntry(BaseModel):
    """Audit trail entry for a decision."""

    entry_id: str = Field(
        ...,
        description="Audit entry identifier"
    )
    decision_id: str = Field(
        ...,
        description="Related decision ID"
    )
    action: str = Field(
        ...,
        description="Action taken"
    )
    actor: str = Field(
        ...,
        description="Who/what performed action"
    )
    reason: str = Field(
        ...,
        description="Reason for action"
    )
    confidence_at_decision: float = Field(
        ...,
        description="Confidence when decision was made"
    )
    outcome: Optional[str] = Field(
        default=None,
        description="Outcome if known"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Entry timestamp"
    )


class ObjectiveFunction(BaseModel):
    """Definition of an objective for optimization."""

    name: str = Field(
        ...,
        description="Objective name"
    )
    objective_type: ObjectiveType = Field(
        ...,
        description="Maximize, minimize, or target"
    )
    target_value: Optional[float] = Field(
        default=None,
        description="Target value if objective_type is TARGET"
    )
    weight: float = Field(
        default=1.0,
        gt=0,
        description="Objective weight"
    )
    uncertainty_penalty: float = Field(
        default=0.0,
        ge=0,
        description="Penalty per unit uncertainty"
    )


class OptimizationResult(BaseModel):
    """Result from multi-objective optimization under uncertainty."""

    optimal_action: str = Field(
        ...,
        description="Optimal action"
    )
    expected_objectives: Dict[str, float] = Field(
        ...,
        description="Expected value for each objective"
    )
    pareto_optimal: bool = Field(
        ...,
        description="Whether solution is Pareto optimal"
    )
    alternatives: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative solutions"
    )
    uncertainty_impact: Dict[str, float] = Field(
        ...,
        description="Impact of uncertainty on each objective"
    )
    robustness_score: float = Field(
        ...,
        description="Solution robustness (0-1)"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )


# ============================================================================
# Decision Engine
# ============================================================================

class DecisionEngine:
    """
    Uncertainty-Aware Decision Engine.

    This class provides decision-making capabilities that incorporate
    prediction uncertainty, enabling confidence-based automation,
    risk-adjusted recommendations, and human-in-the-loop triggers.

    Key capabilities:
    - Confidence thresholds for automated action
    - Risk-adjusted recommendations
    - Human-in-the-loop triggers
    - Multi-objective optimization under uncertainty
    - Decision audit trails

    Attributes:
        config: Decision configuration
        _audit_trail: Decision audit history
        _pending_reviews: Pending human reviews

    Example:
        >>> engine = DecisionEngine(config=DecisionConfig(
        ...     confidence_threshold=0.90,
        ...     risk_tolerance=RiskTolerance.CONSERVATIVE
        ... ))
        >>> decision = engine.make_decision(
        ...     prediction=1.5,
        ...     uncertainty=0.1,
        ...     context={"domain": "emissions"}
        ... )
        >>> if decision.requires_human_review:
        ...     engine.create_review_request(decision)
    """

    def __init__(self, config: Optional[DecisionConfig] = None):
        """
        Initialize decision engine.

        Args:
            config: Decision configuration
        """
        self.config = config or DecisionConfig()
        self._audit_trail: List[DecisionAuditEntry] = []
        self._pending_reviews: Dict[str, HumanReviewRequest] = {}
        self._decision_history: List[Decision] = []

        # Risk multipliers based on tolerance
        self._risk_multipliers = {
            RiskTolerance.AGGRESSIVE: 0.5,
            RiskTolerance.MODERATE: 1.0,
            RiskTolerance.CONSERVATIVE: 1.5,
            RiskTolerance.VERY_CONSERVATIVE: 2.0
        }

        logger.info(
            f"DecisionEngine initialized: threshold={self.config.confidence_threshold}, "
            f"risk_tolerance={self.config.risk_tolerance}"
        )

    def _generate_id(self, prefix: str = "dec") -> str:
        """Generate unique identifier."""
        return f"{prefix}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _calculate_provenance(
        self,
        prediction: float,
        uncertainty: float,
        action: ActionType
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        combined = f"{prediction:.8f}|{uncertainty:.8f}|{action.value}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _classify_confidence(self, score: float) -> str:
        """Classify confidence score into level."""
        if score >= 0.95:
            return "very_high"
        elif score >= 0.85:
            return "high"
        elif score >= 0.70:
            return "medium"
        elif score >= 0.50:
            return "low"
        else:
            return "very_low"

    def _classify_risk(self, risk_score: float) -> str:
        """Classify risk score into level."""
        if risk_score <= 0.2:
            return "low"
        elif risk_score <= 0.5:
            return "medium"
        elif risk_score <= 0.8:
            return "high"
        else:
            return "critical"

    def _compute_confidence_score(
        self,
        uncertainty: float,
        max_uncertainty: float = 1.0
    ) -> float:
        """
        Compute confidence score from uncertainty.

        Args:
            uncertainty: Prediction uncertainty
            max_uncertainty: Maximum expected uncertainty

        Returns:
            Confidence score (0-1)
        """
        # Inverse relationship: lower uncertainty = higher confidence
        normalized = min(uncertainty / max_uncertainty, 1.0)
        confidence = 1 - normalized
        return float(np.clip(confidence, 0, 1))

    def _assess_risk(
        self,
        prediction: float,
        uncertainty: float,
        context: Optional[Dict[str, Any]] = None
    ) -> RiskAssessment:
        """
        Assess risk for a prediction with uncertainty.

        Args:
            prediction: Point prediction
            uncertainty: Prediction uncertainty
            context: Additional context

        Returns:
            RiskAssessment
        """
        context = context or {}

        # Base risk from uncertainty
        uncertainty_risk = min(uncertainty / self.config.uncertainty_threshold, 1.0)

        # Risk multiplier based on tolerance
        risk_multiplier = self._risk_multipliers[self.config.risk_tolerance]

        # Risk factors
        risk_factors = {
            "uncertainty": uncertainty_risk,
            "tolerance_adjusted": uncertainty_risk * risk_multiplier
        }

        # Domain-specific risk factors from context
        if "domain_risk" in context:
            risk_factors["domain"] = context["domain_risk"]

        # Overall risk score
        risk_score = float(np.mean(list(risk_factors.values())))
        risk_score = min(risk_score * risk_multiplier, 1.0)

        # Expected loss calculation
        # Assumes prediction error proportional to uncertainty
        expected_loss = uncertainty * (
            self.config.cost_of_false_positive +
            self.config.cost_of_false_negative
        ) / 2

        # Worst case (95th percentile)
        worst_case_loss = uncertainty * 2 * max(
            self.config.cost_of_false_positive,
            self.config.cost_of_false_negative
        )

        # Recommendations
        recommendations = []
        if risk_score > 0.5:
            recommendations.append("Consider collecting more data to reduce uncertainty.")
        if uncertainty_risk > 0.8:
            recommendations.append("High uncertainty - human review recommended.")
        if self.config.risk_tolerance in [RiskTolerance.CONSERVATIVE, RiskTolerance.VERY_CONSERVATIVE]:
            if risk_score > 0.3:
                recommendations.append("Given conservative risk tolerance, additional verification advised.")

        return RiskAssessment(
            risk_score=risk_score,
            risk_level=self._classify_risk(risk_score),
            risk_factors=risk_factors,
            expected_loss=expected_loss,
            worst_case_loss=worst_case_loss,
            recommendations=recommendations
        )

    def _determine_action(
        self,
        confidence_score: float,
        risk_assessment: RiskAssessment
    ) -> Tuple[ActionType, str]:
        """
        Determine action based on confidence and risk.

        Args:
            confidence_score: Confidence score
            risk_assessment: Risk assessment

        Returns:
            Tuple of (action, justification)
        """
        if confidence_score >= self.config.confidence_threshold:
            if risk_assessment.risk_level in ["low", "medium"]:
                return (
                    ActionType.APPROVE,
                    f"Confidence ({confidence_score:.2%}) meets threshold "
                    f"({self.config.confidence_threshold:.2%}) with acceptable risk."
                )
            else:
                return (
                    ActionType.HUMAN_REVIEW,
                    f"High confidence but elevated risk ({risk_assessment.risk_level}). "
                    "Human review recommended."
                )

        elif confidence_score >= self.config.human_review_threshold:
            return (
                ActionType.HUMAN_REVIEW,
                f"Confidence ({confidence_score:.2%}) below automation threshold "
                f"but above review threshold. Human review required."
            )

        elif confidence_score >= self.config.escalation_threshold:
            return (
                ActionType.DEFER,
                f"Confidence ({confidence_score:.2%}) insufficient for decision. "
                "Deferring for additional information."
            )

        else:
            return (
                ActionType.ESCALATE,
                f"Very low confidence ({confidence_score:.2%}). "
                "Escalation to senior reviewer required."
            )

    def make_decision(
        self,
        prediction: float,
        uncertainty: float,
        context: Optional[Dict[str, Any]] = None,
        max_uncertainty: float = 1.0
    ) -> Decision:
        """
        Make a decision based on prediction and uncertainty.

        Args:
            prediction: Point prediction
            uncertainty: Prediction uncertainty
            context: Additional context for decision
            max_uncertainty: Maximum expected uncertainty for normalization

        Returns:
            Decision with action, justification, and audit trail

        Example:
            >>> decision = engine.make_decision(
            ...     prediction=100.5,
            ...     uncertainty=5.2,
            ...     context={"domain": "emissions", "threshold": 100}
            ... )
            >>> print(f"Action: {decision.action}, Confidence: {decision.confidence.score:.2%}")
        """
        decision_id = self._generate_id("dec")

        # Compute confidence
        confidence_score = self._compute_confidence_score(uncertainty, max_uncertainty)
        confidence = ConfidenceLevel(
            score=confidence_score,
            level=self._classify_confidence(confidence_score),
            threshold_met=confidence_score >= self.config.confidence_threshold
        )

        # Assess risk
        risk_assessment = self._assess_risk(prediction, uncertainty, context)

        # Determine action
        action, justification = self._determine_action(confidence_score, risk_assessment)

        # Determine outcome
        if action in [ActionType.APPROVE, ActionType.REJECT]:
            outcome = DecisionOutcome.AUTOMATED
        elif action in [ActionType.HUMAN_REVIEW, ActionType.ESCALATE]:
            outcome = DecisionOutcome.HUMAN_REQUIRED
        else:
            outcome = DecisionOutcome.PENDING

        # Check if human review required
        requires_human = action in [
            ActionType.HUMAN_REVIEW,
            ActionType.ESCALATE,
            ActionType.DEFER
        ]

        # Consider alternatives
        alternatives = []
        if action == ActionType.DEFER:
            alternatives.append({
                "action": ActionType.APPROVE.value,
                "confidence_required": self.config.confidence_threshold,
                "current_gap": self.config.confidence_threshold - confidence_score
            })
            alternatives.append({
                "action": ActionType.HUMAN_REVIEW.value,
                "recommended_if": "additional context available"
            })

        # Calculate provenance
        provenance_hash = self._calculate_provenance(prediction, uncertainty, action)

        decision = Decision(
            decision_id=decision_id,
            prediction=prediction,
            uncertainty=uncertainty,
            confidence=confidence,
            action=action,
            outcome=outcome,
            risk_assessment=risk_assessment,
            requires_human_review=requires_human,
            justification=justification,
            alternatives=alternatives,
            provenance_hash=provenance_hash
        )

        # Store decision
        self._decision_history.append(decision)

        # Create audit entry
        if self.config.enable_audit_trail:
            self._create_audit_entry(decision, "decision_made", "system")

        logger.info(
            f"Decision made: {decision_id}, action={action.value}, "
            f"confidence={confidence_score:.3f}"
        )

        return decision

    def create_review_request(
        self,
        decision: Decision,
        priority: str = "normal",
        context: Optional[Dict[str, Any]] = None,
        deadline: Optional[datetime] = None
    ) -> HumanReviewRequest:
        """
        Create a human review request for a decision.

        Args:
            decision: Decision requiring review
            priority: Review priority
            context: Additional context
            deadline: Review deadline

        Returns:
            HumanReviewRequest

        Example:
            >>> if decision.requires_human_review:
            ...     request = engine.create_review_request(decision, priority="high")
        """
        request_id = self._generate_id("rev")

        # Determine priority based on risk if not specified
        if priority == "normal" and decision.risk_assessment.risk_level == "critical":
            priority = "urgent"
        elif priority == "normal" and decision.risk_assessment.risk_level == "high":
            priority = "high"

        # Generate detailed confidence explanation
        conf = decision.confidence
        confidence_details = (
            f"Confidence score: {conf.score:.2%} ({conf.level}). "
            f"Uncertainty: {decision.uncertainty:.4f}. "
            f"Threshold for automation: {self.config.confidence_threshold:.2%}. "
            f"Gap: {self.config.confidence_threshold - conf.score:.2%}."
        )

        request = HumanReviewRequest(
            request_id=request_id,
            decision=decision,
            reason=decision.justification,
            priority=priority,
            context=context or {},
            suggested_action=decision.action,
            confidence_details=confidence_details,
            deadline=deadline
        )

        # Store pending review
        self._pending_reviews[request_id] = request

        # Audit entry
        if self.config.enable_audit_trail:
            self._create_audit_entry(
                decision,
                "review_requested",
                "system",
                metadata={"request_id": request_id, "priority": priority}
            )

        logger.info(f"Review request created: {request_id}, priority={priority}")

        return request

    def resolve_review(
        self,
        request_id: str,
        action: ActionType,
        reviewer: str,
        notes: Optional[str] = None
    ) -> Decision:
        """
        Resolve a human review request.

        Args:
            request_id: Review request ID
            action: Action taken by reviewer
            reviewer: Reviewer identifier
            notes: Review notes

        Returns:
            Updated decision

        Example:
            >>> resolved = engine.resolve_review(
            ...     request_id, ActionType.APPROVE, "john.doe", "Approved after verification"
            ... )
        """
        if request_id not in self._pending_reviews:
            raise ValueError(f"Review request not found: {request_id}")

        request = self._pending_reviews.pop(request_id)
        decision = request.decision

        # Update decision
        original_action = decision.action
        decision.action = action
        decision.outcome = DecisionOutcome.AUTOMATED
        decision.requires_human_review = False

        # Audit entry
        if self.config.enable_audit_trail:
            self._create_audit_entry(
                decision,
                "review_resolved",
                reviewer,
                metadata={
                    "original_action": original_action.value,
                    "new_action": action.value,
                    "notes": notes
                }
            )

        logger.info(
            f"Review resolved: {request_id} by {reviewer}, "
            f"action changed from {original_action.value} to {action.value}"
        )

        return decision

    def _create_audit_entry(
        self,
        decision: Decision,
        action: str,
        actor: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DecisionAuditEntry:
        """Create and store audit entry."""
        entry = DecisionAuditEntry(
            entry_id=self._generate_id("aud"),
            decision_id=decision.decision_id,
            action=action,
            actor=actor,
            reason=decision.justification,
            confidence_at_decision=decision.confidence.score,
            metadata=metadata or {},
            provenance_hash=hashlib.sha256(
                f"{decision.decision_id}|{action}|{actor}".encode()
            ).hexdigest()
        )

        self._audit_trail.append(entry)
        return entry

    def optimize_under_uncertainty(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        objectives: List[ObjectiveFunction],
        actions: List[str]
    ) -> OptimizationResult:
        """
        Multi-objective optimization under uncertainty.

        Args:
            predictions: Predictions for each action
            uncertainties: Uncertainties for each action
            objectives: List of objective functions
            actions: List of action names

        Returns:
            OptimizationResult with optimal action

        Example:
            >>> objectives = [
            ...     ObjectiveFunction(name="profit", objective_type=ObjectiveType.MAXIMIZE, weight=1.0),
            ...     ObjectiveFunction(name="risk", objective_type=ObjectiveType.MINIMIZE, weight=0.5)
            ... ]
            >>> result = engine.optimize_under_uncertainty(
            ...     predictions, uncertainties, objectives, ["action_a", "action_b"]
            ... )
        """
        n_actions = len(actions)
        scores = np.zeros(n_actions)
        expected_objectives = {obj.name: {} for obj in objectives}
        uncertainty_impacts = {obj.name: 0.0 for obj in objectives}

        for i, action in enumerate(actions):
            action_score = 0.0
            pred = predictions[i] if i < len(predictions) else 0
            uncert = uncertainties[i] if i < len(uncertainties) else 0

            for obj in objectives:
                # Expected value (prediction)
                expected_value = pred

                # Uncertainty penalty
                penalty = obj.uncertainty_penalty * uncert

                # Objective contribution
                if obj.objective_type == ObjectiveType.MAXIMIZE:
                    contribution = expected_value - penalty
                elif obj.objective_type == ObjectiveType.MINIMIZE:
                    contribution = -(expected_value + penalty)
                else:  # TARGET
                    target = obj.target_value or 0
                    contribution = -abs(expected_value - target) - penalty

                action_score += obj.weight * contribution
                expected_objectives[obj.name][action] = expected_value
                uncertainty_impacts[obj.name] = penalty

            scores[i] = action_score

        # Find optimal action
        optimal_idx = np.argmax(scores)
        optimal_action = actions[optimal_idx]

        # Check Pareto optimality (simplified)
        pareto_optimal = True
        for i in range(n_actions):
            if i != optimal_idx and scores[i] >= scores[optimal_idx]:
                pareto_optimal = False
                break

        # Robustness score (inverse of uncertainty relative impact)
        avg_uncertainty = float(np.mean(uncertainties))
        robustness = 1.0 / (1.0 + avg_uncertainty)

        # Build alternatives
        alternatives = []
        sorted_indices = np.argsort(scores)[::-1]
        for idx in sorted_indices[1:4]:  # Top 3 alternatives
            alternatives.append({
                "action": actions[idx],
                "score": float(scores[idx]),
                "gap_from_optimal": float(scores[optimal_idx] - scores[idx])
            })

        provenance = hashlib.sha256(
            f"{optimal_action}|{scores[optimal_idx]:.8f}".encode()
        ).hexdigest()

        return OptimizationResult(
            optimal_action=optimal_action,
            expected_objectives={
                obj.name: expected_objectives[obj.name].get(optimal_action, 0)
                for obj in objectives
            },
            pareto_optimal=pareto_optimal,
            alternatives=alternatives,
            uncertainty_impact=uncertainty_impacts,
            robustness_score=robustness,
            provenance_hash=provenance
        )

    def get_audit_trail(
        self,
        decision_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[DecisionAuditEntry]:
        """
        Get decision audit trail.

        Args:
            decision_id: Filter by decision ID
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of audit entries
        """
        entries = self._audit_trail

        if decision_id:
            entries = [e for e in entries if e.decision_id == decision_id]

        if start_date:
            entries = [e for e in entries if e.timestamp >= start_date]

        if end_date:
            entries = [e for e in entries if e.timestamp <= end_date]

        return entries

    def get_pending_reviews(self) -> List[HumanReviewRequest]:
        """Get all pending human reviews."""
        return list(self._pending_reviews.values())

    def get_decision_statistics(self) -> Dict[str, Any]:
        """
        Get decision statistics.

        Returns:
            Statistics about decisions made
        """
        if not self._decision_history:
            return {"n_decisions": 0}

        decisions = self._decision_history

        action_counts = {}
        for d in decisions:
            action_counts[d.action.value] = action_counts.get(d.action.value, 0) + 1

        confidences = [d.confidence.score for d in decisions]
        human_review_count = sum(1 for d in decisions if d.requires_human_review)

        return {
            "n_decisions": len(decisions),
            "action_distribution": action_counts,
            "mean_confidence": float(np.mean(confidences)),
            "median_confidence": float(np.median(confidences)),
            "human_review_ratio": human_review_count / len(decisions),
            "pending_reviews": len(self._pending_reviews),
            "audit_entries": len(self._audit_trail)
        }


# ============================================================================
# Unit Tests
# ============================================================================

class TestDecisionEngine:
    """Unit tests for DecisionEngine."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        engine = DecisionEngine()
        assert engine.config.confidence_threshold == 0.90

    def test_make_decision_high_confidence(self):
        """Test decision with high confidence."""
        engine = DecisionEngine(config=DecisionConfig(confidence_threshold=0.90))

        decision = engine.make_decision(
            prediction=1.0,
            uncertainty=0.05,
            max_uncertainty=1.0
        )

        assert decision.confidence.score >= 0.90
        assert decision.action == ActionType.APPROVE

    def test_make_decision_low_confidence(self):
        """Test decision with low confidence."""
        engine = DecisionEngine(config=DecisionConfig(confidence_threshold=0.90))

        decision = engine.make_decision(
            prediction=1.0,
            uncertainty=0.5,
            max_uncertainty=1.0
        )

        assert decision.confidence.score < 0.90
        assert decision.requires_human_review

    def test_create_review_request(self):
        """Test review request creation."""
        engine = DecisionEngine()

        decision = engine.make_decision(
            prediction=1.0,
            uncertainty=0.4,
            max_uncertainty=1.0
        )

        if decision.requires_human_review:
            request = engine.create_review_request(decision)
            assert request.decision.decision_id == decision.decision_id

    def test_resolve_review(self):
        """Test review resolution."""
        engine = DecisionEngine()

        decision = engine.make_decision(
            prediction=1.0,
            uncertainty=0.4,
            max_uncertainty=1.0
        )

        if decision.requires_human_review:
            request = engine.create_review_request(decision)
            resolved = engine.resolve_review(
                request.request_id,
                ActionType.APPROVE,
                "test_reviewer"
            )
            assert resolved.action == ActionType.APPROVE

    def test_risk_assessment(self):
        """Test risk assessment."""
        engine = DecisionEngine(config=DecisionConfig(
            risk_tolerance=RiskTolerance.CONSERVATIVE
        ))

        decision = engine.make_decision(
            prediction=1.0,
            uncertainty=0.3,
            max_uncertainty=1.0
        )

        assert decision.risk_assessment.risk_score >= 0

    def test_optimize_under_uncertainty(self):
        """Test multi-objective optimization."""
        engine = DecisionEngine()

        predictions = np.array([10.0, 8.0, 12.0])
        uncertainties = np.array([1.0, 0.5, 2.0])
        objectives = [
            ObjectiveFunction(
                name="value",
                objective_type=ObjectiveType.MAXIMIZE,
                weight=1.0,
                uncertainty_penalty=0.5
            )
        ]
        actions = ["A", "B", "C"]

        result = engine.optimize_under_uncertainty(
            predictions, uncertainties, objectives, actions
        )

        assert result.optimal_action in actions
        assert result.robustness_score > 0

    def test_audit_trail(self):
        """Test audit trail."""
        engine = DecisionEngine(config=DecisionConfig(enable_audit_trail=True))

        engine.make_decision(prediction=1.0, uncertainty=0.1, max_uncertainty=1.0)

        trail = engine.get_audit_trail()
        assert len(trail) > 0

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        engine = DecisionEngine()

        hash1 = engine._calculate_provenance(1.0, 0.1, ActionType.APPROVE)
        hash2 = engine._calculate_provenance(1.0, 0.1, ActionType.APPROVE)

        assert hash1 == hash2
