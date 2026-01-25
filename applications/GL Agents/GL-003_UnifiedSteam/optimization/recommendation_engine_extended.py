"""
GL-003 UNIFIEDSTEAM - Extended Recommendation Engine

Provides advanced recommendation management capabilities:
- RecommendationAggregator: Aggregates recommendations from all optimizers
- PriorityScoring: ROI, urgency, feasibility-based scoring
- Deduplication: Merges similar recommendations
- Lifecycle Management: Pending -> Acknowledged -> Implemented -> Verified
- Persistence Interface: Save/load recommendations
- Filtering and Search: Query recommendations by various criteria

Every recommendation includes confidence scores and explainability.
"""

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import hashlib
import json
import logging
import time
import uuid

from pydantic import BaseModel, Field, validator

from .recommendation_engine import (
    Recommendation,
    RecommendationType,
    RecommendationPriority,
    RiskCategory,
    OperatorPreference,
    BenefitEstimate,
    RiskAssessment,
    VerificationPlan,
    EscalationPath,
    RankedList,
    RecommendationEngine,
)
from .constraints import UncertaintyConstraints

logger = logging.getLogger(__name__)


# =============================================================================
# Recommendation Status Lifecycle
# =============================================================================


class RecommendationStatus(str, Enum):
    """Status in recommendation lifecycle."""

    PENDING = "pending"  # Newly created, awaiting review
    ACKNOWLEDGED = "acknowledged"  # Operator has seen it
    IN_PROGRESS = "in_progress"  # Implementation started
    IMPLEMENTED = "implemented"  # Changes made to system
    VERIFIED = "verified"  # Benefits confirmed
    REJECTED = "rejected"  # Operator declined
    EXPIRED = "expired"  # No longer relevant
    SUPERSEDED = "superseded"  # Replaced by newer recommendation


class StatusTransition(BaseModel):
    """Record of a status transition."""

    from_status: RecommendationStatus = Field(..., description="Previous status")
    to_status: RecommendationStatus = Field(..., description="New status")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    operator: Optional[str] = Field(default=None, description="Operator who made transition")
    notes: str = Field(default="", description="Transition notes")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# =============================================================================
# Priority Scoring
# =============================================================================


class PriorityScore(BaseModel):
    """Detailed priority score breakdown."""

    recommendation_id: str = Field(..., description="Recommendation ID")
    total_score: float = Field(..., ge=0, le=100, description="Total priority score (0-100)")

    # Component scores
    roi_score: float = Field(default=0.0, ge=0, le=100, description="ROI score")
    urgency_score: float = Field(default=0.0, ge=0, le=100, description="Urgency score")
    feasibility_score: float = Field(default=0.0, ge=0, le=100, description="Feasibility score")
    confidence_score: float = Field(default=0.0, ge=0, le=100, description="Confidence score")
    risk_score: float = Field(default=0.0, ge=0, le=100, description="Risk-adjusted score")

    # Weights used
    weights: Dict[str, float] = Field(
        default_factory=dict, description="Weights used for scoring"
    )

    # Explanation
    explanation: str = Field(default="", description="Score explanation")
    factors: List[str] = Field(default_factory=list, description="Contributing factors")


class PriorityScoringEngine:
    """
    Calculates priority scores for recommendations.

    Scoring considers:
    - ROI: Return on investment (cost savings vs implementation cost)
    - Urgency: Time sensitivity and safety implications
    - Feasibility: Implementation complexity and resource availability
    - Confidence: Data quality and model confidence
    - Risk: Implementation risk adjusted
    """

    DEFAULT_WEIGHTS = {
        "roi": 0.30,
        "urgency": 0.25,
        "feasibility": 0.20,
        "confidence": 0.15,
        "risk": 0.10,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        labor_rate: float = 75.0,
        implementation_hours: Dict[str, float] = None,
    ) -> None:
        """
        Initialize priority scoring engine.

        Args:
            weights: Custom weights for score components
            labor_rate: Labor rate for implementation cost calculation
            implementation_hours: Estimated hours by recommendation type
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.labor_rate = labor_rate
        self.implementation_hours = implementation_hours or {
            RecommendationType.SETPOINT_CHANGE.value: 0.5,
            RecommendationType.EQUIPMENT_ACTION.value: 2.0,
            RecommendationType.MAINTENANCE.value: 4.0,
            RecommendationType.OPERATIONAL.value: 1.0,
            RecommendationType.ALARM.value: 0.25,
            RecommendationType.INFORMATIONAL.value: 0.0,
        }

    def calculate_score(
        self,
        recommendation: Recommendation,
        context: Optional[Dict[str, Any]] = None,
    ) -> PriorityScore:
        """
        Calculate comprehensive priority score.

        Args:
            recommendation: Recommendation to score
            context: Additional context (resource availability, etc.)

        Returns:
            PriorityScore with detailed breakdown
        """
        factors = []

        # Calculate ROI score
        roi_score, roi_factors = self._calculate_roi_score(recommendation)
        factors.extend(roi_factors)

        # Calculate urgency score
        urgency_score, urgency_factors = self._calculate_urgency_score(recommendation)
        factors.extend(urgency_factors)

        # Calculate feasibility score
        feasibility_score, feasibility_factors = self._calculate_feasibility_score(
            recommendation, context
        )
        factors.extend(feasibility_factors)

        # Calculate confidence score
        confidence_score = recommendation.expected_benefit.confidence * 100
        if confidence_score >= 90:
            factors.append("High confidence (90%+)")
        elif confidence_score < 70:
            factors.append("Low confidence - verify data")

        # Calculate risk-adjusted score (lower risk = higher score)
        risk_score = 100 - recommendation.risk_assessment.risk_score
        if risk_score < 50:
            factors.append("Elevated implementation risk")

        # Calculate weighted total
        total_score = (
            roi_score * self.weights["roi"] +
            urgency_score * self.weights["urgency"] +
            feasibility_score * self.weights["feasibility"] +
            confidence_score * self.weights["confidence"] +
            risk_score * self.weights["risk"]
        )

        # Generate explanation
        explanation = self._generate_explanation(
            total_score, roi_score, urgency_score, feasibility_score
        )

        return PriorityScore(
            recommendation_id=recommendation.recommendation_id,
            total_score=round(total_score, 1),
            roi_score=round(roi_score, 1),
            urgency_score=round(urgency_score, 1),
            feasibility_score=round(feasibility_score, 1),
            confidence_score=round(confidence_score, 1),
            risk_score=round(risk_score, 1),
            weights=self.weights,
            explanation=explanation,
            factors=factors,
        )

    def _calculate_roi_score(
        self,
        recommendation: Recommendation,
    ) -> Tuple[float, List[str]]:
        """Calculate ROI-based score."""
        factors = []
        benefit = recommendation.expected_benefit

        # Calculate implementation cost
        impl_hours = self.implementation_hours.get(
            recommendation.recommendation_type.value, 1.0
        )
        impl_cost = impl_hours * self.labor_rate

        # Calculate annual benefit
        annual_benefit = benefit.annual_cost_savings

        # ROI calculation
        if impl_cost > 0:
            roi = annual_benefit / impl_cost
            if roi > 100:
                score = 100
                factors.append(f"Excellent ROI ({roi:.0f}x)")
            elif roi > 20:
                score = 80 + (roi - 20) * 0.25
                factors.append(f"Strong ROI ({roi:.0f}x)")
            elif roi > 5:
                score = 50 + (roi - 5) * 2
                factors.append(f"Good ROI ({roi:.0f}x)")
            elif roi > 1:
                score = 25 + (roi - 1) * 6.25
                factors.append(f"Positive ROI ({roi:.1f}x)")
            else:
                score = roi * 25
                factors.append(f"Low ROI ({roi:.1f}x)")
        else:
            # No implementation cost - score based on benefit
            if annual_benefit > 100000:
                score = 100
            elif annual_benefit > 10000:
                score = 70 + (annual_benefit - 10000) / 3000
            else:
                score = annual_benefit / 10000 * 70

        # Bonus for CO2 reduction
        if benefit.annual_co2e_reduction_tons > 100:
            score = min(100, score + 10)
            factors.append(f"Significant CO2 reduction ({benefit.annual_co2e_reduction_tons:.0f} tons/yr)")

        return min(100, score), factors

    def _calculate_urgency_score(
        self,
        recommendation: Recommendation,
    ) -> Tuple[float, List[str]]:
        """Calculate urgency-based score."""
        factors = []

        # Base score from priority
        priority_scores = {
            RecommendationPriority.CRITICAL: 100,
            RecommendationPriority.HIGH: 80,
            RecommendationPriority.MEDIUM: 50,
            RecommendationPriority.LOW: 25,
            RecommendationPriority.INFORMATIONAL: 10,
        }
        score = priority_scores.get(recommendation.priority, 50)

        if recommendation.priority == RecommendationPriority.CRITICAL:
            factors.append("CRITICAL priority - immediate action required")
        elif recommendation.priority == RecommendationPriority.HIGH:
            factors.append("HIGH priority - act within 1 hour")

        # Check for safety implications
        risk = recommendation.risk_assessment
        for r in risk.risks:
            if r.get("category") == RiskCategory.SAFETY.value:
                score = min(100, score + 20)
                factors.append("Safety implications")
                break

        # Check age of recommendation
        age_hours = (datetime.now(timezone.utc) - recommendation.timestamp).total_seconds() / 3600
        if age_hours > 24:
            score = min(100, score + 10)
            factors.append("Recommendation aging (>24h)")

        return score, factors

    def _calculate_feasibility_score(
        self,
        recommendation: Recommendation,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, List[str]]:
        """Calculate feasibility-based score."""
        factors = []
        score = 80  # Base score

        # Reduce for complex recommendations
        if recommendation.recommendation_type == RecommendationType.MAINTENANCE:
            score -= 20
            factors.append("Maintenance requires scheduling")
        elif recommendation.recommendation_type == RecommendationType.EQUIPMENT_ACTION:
            score -= 10
            factors.append("Equipment action required")

        # Check if operator confirmation required
        if recommendation.risk_assessment.requires_operator_confirmation:
            score -= 10
            factors.append("Requires operator confirmation")

        if recommendation.risk_assessment.requires_engineering_review:
            score -= 20
            factors.append("Requires engineering review")

        # Context-based adjustments
        if context:
            if context.get("maintenance_mode", False):
                score -= 30
                factors.append("System in maintenance mode")
            if context.get("peak_hours", False):
                score -= 15
                factors.append("Peak hours - limited changes")
            if context.get("staffing_low", False):
                score -= 20
                factors.append("Limited staffing")

        return max(0, score), factors

    def _generate_explanation(
        self,
        total: float,
        roi: float,
        urgency: float,
        feasibility: float,
    ) -> str:
        """Generate human-readable explanation."""
        if total >= 80:
            return "High priority recommendation with strong ROI and feasibility"
        elif total >= 60:
            return "Good recommendation worth implementing"
        elif total >= 40:
            return "Moderate priority - consider when resources available"
        elif total >= 20:
            return "Lower priority - implement opportunistically"
        else:
            return "Low priority - for reference only"


# =============================================================================
# Explainability
# =============================================================================


class ExplainabilityInfo(BaseModel):
    """Explainability information for a recommendation."""

    recommendation_id: str = Field(..., description="Recommendation ID")
    summary: str = Field(..., description="Human-readable summary")

    # Data sources
    data_sources: List[str] = Field(
        default_factory=list, description="Data sources used"
    )
    data_freshness_minutes: int = Field(
        default=0, description="Age of newest data (minutes)"
    )

    # Calculation breakdown
    calculation_steps: List[Dict[str, Any]] = Field(
        default_factory=list, description="Calculation steps with values"
    )

    # Assumptions
    assumptions: List[str] = Field(
        default_factory=list, description="Key assumptions"
    )

    # Sensitivity
    sensitivity_analysis: Dict[str, float] = Field(
        default_factory=dict, description="Sensitivity to key inputs"
    )

    # Alternative scenarios
    alternatives_considered: List[str] = Field(
        default_factory=list, description="Alternative actions considered"
    )

    # Confidence factors
    confidence_factors: Dict[str, float] = Field(
        default_factory=dict, description="Factors affecting confidence"
    )

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# =============================================================================
# Recommendation Aggregator
# =============================================================================


class RecommendationAggregator:
    """
    Aggregates recommendations from multiple optimizers.

    Features:
    - Collects recommendations from all optimizer sources
    - Deduplicates similar recommendations
    - Merges related recommendations
    - Resolves conflicts between recommendations
    - Maintains global recommendation state
    """

    def __init__(
        self,
        scoring_engine: Optional[PriorityScoringEngine] = None,
        deduplication_threshold: float = 0.8,
    ) -> None:
        """
        Initialize aggregator.

        Args:
            scoring_engine: Priority scoring engine
            deduplication_threshold: Similarity threshold for deduplication
        """
        self.scoring_engine = scoring_engine or PriorityScoringEngine()
        self.deduplication_threshold = deduplication_threshold
        self.recommendations: Dict[str, Recommendation] = {}
        self.scores: Dict[str, PriorityScore] = {}
        self.status_history: Dict[str, List[StatusTransition]] = {}

        logger.info("RecommendationAggregator initialized")

    def add_recommendation(
        self,
        recommendation: Recommendation,
        source: str = "unknown",
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool]:
        """
        Add a recommendation to the aggregator.

        Args:
            recommendation: Recommendation to add
            source: Source optimizer name
            context: Additional context

        Returns:
            Tuple of (recommendation_id, was_merged)
        """
        # Check for duplicates
        duplicate_id = self._find_duplicate(recommendation)
        if duplicate_id:
            merged = self._merge_recommendations(duplicate_id, recommendation)
            logger.info(f"Merged recommendation into {duplicate_id}")
            return duplicate_id, True

        # Add new recommendation
        self.recommendations[recommendation.recommendation_id] = recommendation

        # Calculate priority score
        score = self.scoring_engine.calculate_score(recommendation, context)
        self.scores[recommendation.recommendation_id] = score

        # Initialize status history
        self.status_history[recommendation.recommendation_id] = [
            StatusTransition(
                from_status=RecommendationStatus.PENDING,
                to_status=RecommendationStatus.PENDING,
                notes=f"Created from {source}",
            )
        ]

        logger.info(
            f"Added recommendation {recommendation.recommendation_id}: "
            f"score={score.total_score:.1f}"
        )

        return recommendation.recommendation_id, False

    def _find_duplicate(self, recommendation: Recommendation) -> Optional[str]:
        """Find duplicate or very similar recommendation."""
        for rec_id, existing in self.recommendations.items():
            similarity = self._calculate_similarity(existing, recommendation)
            if similarity >= self.deduplication_threshold:
                return rec_id
        return None

    def _calculate_similarity(
        self,
        rec1: Recommendation,
        rec2: Recommendation,
    ) -> float:
        """Calculate similarity between two recommendations."""
        score = 0.0

        # Same type
        if rec1.recommendation_type == rec2.recommendation_type:
            score += 0.3

        # Same target equipment
        common_equipment = set(rec1.target_equipment) & set(rec2.target_equipment)
        if common_equipment:
            score += 0.3 * len(common_equipment) / max(
                len(rec1.target_equipment), len(rec2.target_equipment), 1
            )

        # Similar action (by checking for common words)
        words1 = set(rec1.action.lower().split())
        words2 = set(rec2.action.lower().split())
        common_words = words1 & words2
        if common_words:
            score += 0.2 * len(common_words) / max(len(words1), len(words2), 1)

        # Similar priority
        if rec1.priority == rec2.priority:
            score += 0.2

        return score

    def _merge_recommendations(
        self,
        existing_id: str,
        new_rec: Recommendation,
    ) -> Recommendation:
        """Merge new recommendation into existing one."""
        existing = self.recommendations[existing_id]

        # Update benefit estimate (take better)
        if new_rec.expected_benefit.cost_savings_per_hr > existing.expected_benefit.cost_savings_per_hr:
            existing.expected_benefit = new_rec.expected_benefit

        # Update confidence (take higher)
        if new_rec.expected_benefit.confidence > existing.expected_benefit.confidence:
            existing.expected_benefit.confidence = new_rec.expected_benefit.confidence

        # Merge target equipment
        existing.target_equipment = list(
            set(existing.target_equipment) | set(new_rec.target_equipment)
        )

        # Update supporting data
        existing.supporting_data.update(new_rec.supporting_data)

        return existing

    def get_ranked_recommendations(
        self,
        preference: OperatorPreference = OperatorPreference.BALANCED,
        max_count: int = 20,
        min_score: float = 0.0,
        statuses: Optional[List[RecommendationStatus]] = None,
    ) -> List[Tuple[Recommendation, PriorityScore]]:
        """
        Get ranked list of recommendations.

        Args:
            preference: Operator preference for ranking
            max_count: Maximum recommendations to return
            min_score: Minimum score threshold
            statuses: Filter by statuses (default: PENDING)

        Returns:
            List of (recommendation, score) tuples, sorted by score
        """
        statuses = statuses or [RecommendationStatus.PENDING]

        # Filter by status
        filtered = []
        for rec_id, rec in self.recommendations.items():
            current_status = self._get_current_status(rec_id)
            if current_status in statuses:
                score = self.scores.get(rec_id)
                if score and score.total_score >= min_score:
                    filtered.append((rec, score))

        # Sort by score
        filtered.sort(key=lambda x: x[1].total_score, reverse=True)

        return filtered[:max_count]

    def _get_current_status(self, rec_id: str) -> RecommendationStatus:
        """Get current status of a recommendation."""
        history = self.status_history.get(rec_id, [])
        if history:
            return history[-1].to_status
        return RecommendationStatus.PENDING


# =============================================================================
# Lifecycle Manager
# =============================================================================


class RecommendationLifecycleManager:
    """
    Manages the lifecycle of recommendations.

    States:
    - PENDING: New recommendation awaiting review
    - ACKNOWLEDGED: Operator has reviewed
    - IN_PROGRESS: Implementation started
    - IMPLEMENTED: Changes applied to system
    - VERIFIED: Benefits confirmed
    - REJECTED: Operator declined
    - EXPIRED: No longer relevant
    - SUPERSEDED: Replaced by newer recommendation
    """

    # Valid transitions
    VALID_TRANSITIONS = {
        RecommendationStatus.PENDING: [
            RecommendationStatus.ACKNOWLEDGED,
            RecommendationStatus.REJECTED,
            RecommendationStatus.EXPIRED,
        ],
        RecommendationStatus.ACKNOWLEDGED: [
            RecommendationStatus.IN_PROGRESS,
            RecommendationStatus.REJECTED,
            RecommendationStatus.EXPIRED,
        ],
        RecommendationStatus.IN_PROGRESS: [
            RecommendationStatus.IMPLEMENTED,
            RecommendationStatus.REJECTED,
        ],
        RecommendationStatus.IMPLEMENTED: [
            RecommendationStatus.VERIFIED,
            RecommendationStatus.REJECTED,
        ],
        RecommendationStatus.VERIFIED: [],  # Final state
        RecommendationStatus.REJECTED: [],  # Final state
        RecommendationStatus.EXPIRED: [],  # Final state
        RecommendationStatus.SUPERSEDED: [],  # Final state
    }

    def __init__(
        self,
        aggregator: RecommendationAggregator,
        expiration_hours: int = 24,
    ) -> None:
        """
        Initialize lifecycle manager.

        Args:
            aggregator: Recommendation aggregator
            expiration_hours: Hours before pending recommendations expire
        """
        self.aggregator = aggregator
        self.expiration_hours = expiration_hours

    def transition(
        self,
        rec_id: str,
        new_status: RecommendationStatus,
        operator: str,
        notes: str = "",
    ) -> Tuple[bool, str]:
        """
        Transition recommendation to new status.

        Args:
            rec_id: Recommendation ID
            new_status: Target status
            operator: Operator making transition
            notes: Transition notes

        Returns:
            Tuple of (success, message)
        """
        if rec_id not in self.aggregator.recommendations:
            return False, f"Recommendation {rec_id} not found"

        current_status = self.aggregator._get_current_status(rec_id)

        # Check valid transition
        valid_targets = self.VALID_TRANSITIONS.get(current_status, [])
        if new_status not in valid_targets:
            return False, (
                f"Invalid transition from {current_status.value} "
                f"to {new_status.value}"
            )

        # Record transition
        transition = StatusTransition(
            from_status=current_status,
            to_status=new_status,
            operator=operator,
            notes=notes,
        )

        # Generate provenance hash
        hash_data = (
            f"{rec_id}{current_status.value}{new_status.value}"
            f"{transition.timestamp.isoformat()}{operator}"
        )
        transition.provenance_hash = hashlib.sha256(hash_data.encode()).hexdigest()

        self.aggregator.status_history[rec_id].append(transition)

        logger.info(
            f"Recommendation {rec_id}: {current_status.value} -> {new_status.value} "
            f"by {operator}"
        )

        return True, f"Transitioned to {new_status.value}"

    def expire_old_recommendations(self) -> List[str]:
        """
        Expire recommendations that have been pending too long.

        Returns:
            List of expired recommendation IDs
        """
        expired = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.expiration_hours)

        for rec_id, rec in self.aggregator.recommendations.items():
            current_status = self.aggregator._get_current_status(rec_id)
            if current_status == RecommendationStatus.PENDING:
                if rec.timestamp < cutoff:
                    success, _ = self.transition(
                        rec_id,
                        RecommendationStatus.EXPIRED,
                        "system",
                        f"Auto-expired after {self.expiration_hours} hours"
                    )
                    if success:
                        expired.append(rec_id)

        if expired:
            logger.info(f"Expired {len(expired)} old recommendations")

        return expired


# =============================================================================
# Persistence Interface
# =============================================================================


class RecommendationPersistence:
    """
    Persistence interface for recommendations.

    Supports:
    - JSON file storage
    - In-memory caching
    - Async I/O (future)
    """

    def __init__(self, storage_path: Optional[str] = None) -> None:
        """
        Initialize persistence.

        Args:
            storage_path: Path to storage file
        """
        self.storage_path = storage_path
        self._cache: Dict[str, Any] = {}

    def save_recommendation(
        self,
        recommendation: Recommendation,
        score: Optional[PriorityScore] = None,
        history: Optional[List[StatusTransition]] = None,
    ) -> str:
        """
        Save a recommendation.

        Args:
            recommendation: Recommendation to save
            score: Priority score
            history: Status history

        Returns:
            Recommendation ID
        """
        data = {
            "recommendation": recommendation.dict(),
            "score": score.dict() if score else None,
            "history": [t.dict() for t in history] if history else [],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        self._cache[recommendation.recommendation_id] = data

        if self.storage_path:
            self._write_to_file()

        return recommendation.recommendation_id

    def load_recommendation(
        self,
        rec_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Load a recommendation.

        Args:
            rec_id: Recommendation ID

        Returns:
            Recommendation data or None if not found
        """
        if rec_id in self._cache:
            return self._cache[rec_id]

        if self.storage_path:
            self._read_from_file()
            return self._cache.get(rec_id)

        return None

    def search_recommendations(
        self,
        filters: Dict[str, Any],
    ) -> List[str]:
        """
        Search recommendations by filters.

        Args:
            filters: Search filters (type, priority, date_range, etc.)

        Returns:
            List of matching recommendation IDs
        """
        matches = []

        for rec_id, data in self._cache.items():
            rec_data = data.get("recommendation", {})
            match = True

            if "type" in filters:
                if rec_data.get("recommendation_type") != filters["type"]:
                    match = False

            if "priority" in filters:
                if rec_data.get("priority") != filters["priority"]:
                    match = False

            if "min_score" in filters:
                score_data = data.get("score", {})
                if score_data.get("total_score", 0) < filters["min_score"]:
                    match = False

            if "equipment" in filters:
                if filters["equipment"] not in rec_data.get("target_equipment", []):
                    match = False

            if match:
                matches.append(rec_id)

        return matches

    def _write_to_file(self) -> None:
        """Write cache to file."""
        if not self.storage_path:
            return

        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self._cache, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to write recommendations: {e}")

    def _read_from_file(self) -> None:
        """Read cache from file."""
        if not self.storage_path:
            return

        try:
            with open(self.storage_path, 'r') as f:
                self._cache = json.load(f)
        except FileNotFoundError:
            self._cache = {}
        except Exception as e:
            logger.error(f"Failed to read recommendations: {e}")
            self._cache = {}


# =============================================================================
# Enhanced Recommendation Engine
# =============================================================================


class EnhancedRecommendationEngine(RecommendationEngine):
    """
    Enhanced recommendation engine with lifecycle management.

    Extends base RecommendationEngine with:
    - Recommendation aggregation from multiple sources
    - Priority scoring with ROI/urgency/feasibility
    - Deduplication and merging
    - Full lifecycle management
    - Persistence
    - Explainability
    """

    def __init__(
        self,
        uncertainty_constraints: Optional[UncertaintyConstraints] = None,
        operating_hours: int = 8000,
        co2_cost_per_ton: float = 0.0,
        storage_path: Optional[str] = None,
        expiration_hours: int = 24,
    ) -> None:
        """
        Initialize enhanced recommendation engine.

        Args:
            uncertainty_constraints: Constraints for uncertainty handling
            operating_hours: Annual operating hours
            co2_cost_per_ton: CO2 cost for emissions valuation
            storage_path: Path for persistence
            expiration_hours: Hours before recommendations expire
        """
        super().__init__(
            uncertainty_constraints=uncertainty_constraints,
            operating_hours=operating_hours,
            co2_cost_per_ton=co2_cost_per_ton,
        )

        # Initialize components
        self.scoring_engine = PriorityScoringEngine()
        self.aggregator = RecommendationAggregator(
            scoring_engine=self.scoring_engine
        )
        self.lifecycle_manager = RecommendationLifecycleManager(
            aggregator=self.aggregator,
            expiration_hours=expiration_hours,
        )
        self.persistence = RecommendationPersistence(storage_path)

        logger.info("EnhancedRecommendationEngine initialized")

    def process_optimization_result(
        self,
        optimization_result: Any,
        context: Dict[str, Any],
        source: str = "unknown",
    ) -> Tuple[Recommendation, PriorityScore, ExplainabilityInfo]:
        """
        Process an optimization result into a scored, explainable recommendation.

        Args:
            optimization_result: Result from an optimizer
            context: Additional context
            source: Source optimizer name

        Returns:
            Tuple of (Recommendation, PriorityScore, ExplainabilityInfo)
        """
        start_time = time.perf_counter()

        # Create recommendation
        recommendation = self.package_recommendation(optimization_result, context)

        # Add to aggregator
        rec_id, was_merged = self.aggregator.add_recommendation(
            recommendation, source, context
        )

        # Get score
        score = self.aggregator.scores[rec_id]

        # Create explainability info
        explainability = self._create_explainability(
            recommendation, optimization_result, context
        )

        # Persist
        self.persistence.save_recommendation(
            recommendation,
            score,
            self.aggregator.status_history.get(rec_id, [])
        )

        computation_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Processed recommendation {rec_id}: score={score.total_score:.1f}, "
            f"merged={was_merged}, in {computation_time:.1f}ms"
        )

        return recommendation, score, explainability

    def _create_explainability(
        self,
        recommendation: Recommendation,
        optimization_result: Any,
        context: Dict[str, Any],
    ) -> ExplainabilityInfo:
        """Create explainability information for a recommendation."""
        result_type = type(optimization_result).__name__

        # Collect data sources
        data_sources = ["Process historian", "Real-time sensors"]
        if hasattr(optimization_result, 'data_sources'):
            data_sources = optimization_result.data_sources

        # Calculation steps
        calculation_steps = []
        if hasattr(optimization_result, 'cost_savings_per_hour'):
            calculation_steps.append({
                "step": "Cost savings calculation",
                "inputs": {"steam_cost": context.get("steam_cost_per_klb", 10.0)},
                "output": optimization_result.cost_savings_per_hour,
            })

        # Assumptions
        assumptions = recommendation.expected_benefit.assumptions

        # Generate summary
        summary = (
            f"This recommendation is based on {result_type} analysis with "
            f"{recommendation.expected_benefit.confidence:.0%} confidence. "
            f"Expected annual savings: ${recommendation.expected_benefit.annual_cost_savings:,.0f}. "
            f"{recommendation.rationale}"
        )

        # Provenance hash
        hash_data = (
            f"{recommendation.recommendation_id}"
            f"{result_type}"
            f"{recommendation.timestamp.isoformat()}"
        )
        provenance = hashlib.sha256(hash_data.encode()).hexdigest()

        return ExplainabilityInfo(
            recommendation_id=recommendation.recommendation_id,
            summary=summary,
            data_sources=data_sources,
            data_freshness_minutes=int(context.get("data_age_minutes", 5)),
            calculation_steps=calculation_steps,
            assumptions=assumptions,
            sensitivity_analysis=context.get("sensitivity", {}),
            alternatives_considered=[],
            confidence_factors={
                "data_quality": context.get("data_quality", 0.9),
                "model_accuracy": context.get("model_accuracy", 0.85),
            },
            provenance_hash=provenance,
        )

    def acknowledge(
        self,
        rec_id: str,
        operator: str,
        notes: str = "",
    ) -> Tuple[bool, str]:
        """
        Acknowledge a recommendation.

        Args:
            rec_id: Recommendation ID
            operator: Operator name
            notes: Optional notes

        Returns:
            Tuple of (success, message)
        """
        return self.lifecycle_manager.transition(
            rec_id, RecommendationStatus.ACKNOWLEDGED, operator, notes
        )

    def start_implementation(
        self,
        rec_id: str,
        operator: str,
        notes: str = "",
    ) -> Tuple[bool, str]:
        """Start implementation of a recommendation."""
        return self.lifecycle_manager.transition(
            rec_id, RecommendationStatus.IN_PROGRESS, operator, notes
        )

    def mark_implemented(
        self,
        rec_id: str,
        operator: str,
        notes: str = "",
    ) -> Tuple[bool, str]:
        """Mark recommendation as implemented."""
        return self.lifecycle_manager.transition(
            rec_id, RecommendationStatus.IMPLEMENTED, operator, notes
        )

    def verify(
        self,
        rec_id: str,
        operator: str,
        actual_savings: float,
        notes: str = "",
    ) -> Tuple[bool, str]:
        """
        Verify recommendation benefits were achieved.

        Args:
            rec_id: Recommendation ID
            operator: Operator name
            actual_savings: Actual savings realized
            notes: Verification notes

        Returns:
            Tuple of (success, message)
        """
        verification_notes = (
            f"Verified by {operator}. Actual savings: ${actual_savings:.2f}/hr. {notes}"
        )
        return self.lifecycle_manager.transition(
            rec_id, RecommendationStatus.VERIFIED, operator, verification_notes
        )

    def reject(
        self,
        rec_id: str,
        operator: str,
        reason: str,
    ) -> Tuple[bool, str]:
        """Reject a recommendation."""
        return self.lifecycle_manager.transition(
            rec_id, RecommendationStatus.REJECTED, operator, reason
        )

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Get summary for operator dashboard.

        Returns:
            Dashboard summary dictionary
        """
        pending = []
        acknowledged = []
        in_progress = []
        total_potential_savings = 0.0

        for rec_id, rec in self.aggregator.recommendations.items():
            status = self.aggregator._get_current_status(rec_id)
            score = self.aggregator.scores.get(rec_id)

            if status == RecommendationStatus.PENDING:
                pending.append((rec, score))
                total_potential_savings += rec.expected_benefit.cost_savings_per_hr
            elif status == RecommendationStatus.ACKNOWLEDGED:
                acknowledged.append((rec, score))
            elif status == RecommendationStatus.IN_PROGRESS:
                in_progress.append((rec, score))

        # Sort by score
        pending.sort(key=lambda x: x[1].total_score if x[1] else 0, reverse=True)

        return {
            "pending_count": len(pending),
            "acknowledged_count": len(acknowledged),
            "in_progress_count": len(in_progress),
            "total_recommendations": len(self.aggregator.recommendations),
            "total_potential_savings_per_hr": round(total_potential_savings, 2),
            "top_pending": [
                {
                    "id": r.recommendation_id,
                    "action": r.action,
                    "priority": r.priority.value,
                    "score": s.total_score if s else 0,
                    "savings_per_hr": r.expected_benefit.cost_savings_per_hr,
                }
                for r, s in pending[:5]
            ],
        }
