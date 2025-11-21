# -*- coding: utf-8 -*-
"""
Feedback Data Models for GL-002 BoilerEfficiencyOptimizer

This module defines Pydantic models for feedback collection, storage, and analysis.
All models include validation rules and SHA-256 provenance tracking.

Example:
    >>> feedback = OptimizationFeedback(
    ...     optimization_id="opt_123",
    ...     rating=5,
    ...     comment="Excellent savings!",
    ...     actual_savings=2500.0
    ... )
    >>> assert feedback.rating >= 1 and feedback.rating <= 5
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime
from enum import Enum
import hashlib


class FeedbackRating(int, Enum):
    """Star rating enumeration (1-5 stars)."""
    ONE_STAR = 1
    TWO_STARS = 2
    THREE_STARS = 3
    FOUR_STARS = 4
    FIVE_STARS = 5


class FeedbackCategory(str, Enum):
    """Feedback category types."""
    ACCURACY = "accuracy"
    USABILITY = "usability"
    PERFORMANCE = "performance"
    SAVINGS = "savings"
    RELIABILITY = "reliability"
    OTHER = "other"


class OptimizationFeedback(BaseModel):
    """
    User feedback for a specific optimization recommendation.

    This model captures user satisfaction, comments, and actual results
    to enable continuous improvement of optimization algorithms.

    Attributes:
        optimization_id: Unique identifier for the optimization
        rating: User satisfaction rating (1-5 stars)
        comment: Optional detailed feedback comment
        actual_savings: Actual energy savings achieved (kWh)
        predicted_savings: Predicted savings from optimization (kWh)
        category: Feedback category for classification
        user_id: Identifier of the user providing feedback
        timestamp: When feedback was submitted
        metadata: Additional context (boiler_id, plant_id, etc.)
    """

    optimization_id: str = Field(..., description="Unique optimization identifier")
    rating: int = Field(..., ge=1, le=5, description="Satisfaction rating (1-5 stars)")
    comment: Optional[str] = Field(None, max_length=2000, description="Detailed feedback")
    actual_savings: Optional[float] = Field(None, ge=0, description="Actual savings in kWh")
    predicted_savings: Optional[float] = Field(None, ge=0, description="Predicted savings in kWh")
    category: FeedbackCategory = Field(FeedbackCategory.OTHER, description="Feedback category")
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Submission time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    # Calculated fields
    savings_accuracy: Optional[float] = Field(None, description="Prediction accuracy percentage")
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash for audit trail")

    @validator('rating')
    def validate_rating(cls, v: int) -> int:
        """Ensure rating is within 1-5 range."""
        if v < 1 or v > 5:
            raise ValueError(f"Rating must be between 1 and 5, got {v}")
        return v

    @validator('comment')
    def validate_comment(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize comment input."""
        if v is not None:
            # Remove excessive whitespace
            v = " ".join(v.split())
            # Ensure minimum length if provided
            if len(v) < 3:
                raise ValueError("Comment must be at least 3 characters if provided")
        return v

    @root_validator
    def calculate_savings_accuracy(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate savings prediction accuracy if both values available."""
        actual = values.get('actual_savings')
        predicted = values.get('predicted_savings')

        if actual is not None and predicted is not None and predicted > 0:
            # Calculate percentage accuracy (100% = perfect prediction)
            accuracy = 100 * (1 - abs(actual - predicted) / predicted)
            values['savings_accuracy'] = round(accuracy, 2)

        return values

    @root_validator
    def calculate_provenance(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate SHA-256 hash for audit trail."""
        # Create deterministic string for hashing
        provenance_data = f"{values['optimization_id']}|{values['rating']}|{values['user_id']}|{values['timestamp'].isoformat()}"
        values['provenance_hash'] = hashlib.sha256(provenance_data.encode()).hexdigest()
        return values

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FeedbackStats(BaseModel):
    """
    Aggregated feedback statistics.

    Provides summary metrics for monitoring user satisfaction over time.
    """

    total_feedback_count: int = Field(..., ge=0, description="Total feedback submissions")
    average_rating: float = Field(..., ge=1.0, le=5.0, description="Average star rating")
    rating_distribution: Dict[int, int] = Field(..., description="Count per rating (1-5)")
    average_savings_accuracy: Optional[float] = Field(None, description="Average prediction accuracy %")
    nps_score: Optional[float] = Field(None, ge=-100, le=100, description="Net Promoter Score")

    # Time-based stats
    period_start: datetime = Field(..., description="Statistics period start")
    period_end: datetime = Field(..., description="Statistics period end")

    # Category breakdown
    category_distribution: Dict[str, int] = Field(default_factory=dict, description="Feedback per category")

    @validator('rating_distribution')
    def validate_rating_distribution(cls, v: Dict[int, int]) -> Dict[int, int]:
        """Ensure all rating keys are valid (1-5)."""
        for rating in v.keys():
            if rating < 1 or rating > 5:
                raise ValueError(f"Invalid rating key: {rating}")
        return v

    @root_validator
    def calculate_nps(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Net Promoter Score (NPS).

        NPS = (% Promoters [5 stars] - % Detractors [1-3 stars])
        Range: -100 to +100
        """
        distribution = values.get('rating_distribution', {})
        total = values.get('total_feedback_count', 0)

        if total > 0:
            promoters = distribution.get(5, 0) + distribution.get(4, 0)  # 4-5 stars
            detractors = distribution.get(1, 0) + distribution.get(2, 0) + distribution.get(3, 0)  # 1-3 stars

            nps = ((promoters - detractors) / total) * 100
            values['nps_score'] = round(nps, 2)

        return values


class SatisfactionTrend(BaseModel):
    """
    Time-series data for satisfaction tracking.

    Enables trend analysis and anomaly detection in user satisfaction.
    """

    date: datetime = Field(..., description="Date of measurement")
    average_rating: float = Field(..., ge=1.0, le=5.0, description="Average rating for date")
    feedback_count: int = Field(..., ge=0, description="Number of feedback submissions")
    nps_score: Optional[float] = Field(None, description="NPS for date")

    # Moving averages for smoothing
    ma_7day: Optional[float] = Field(None, description="7-day moving average")
    ma_30day: Optional[float] = Field(None, description="30-day moving average")

    # Anomaly detection
    is_anomaly: bool = Field(False, description="Statistical anomaly detected")
    anomaly_score: Optional[float] = Field(None, description="Anomaly severity score")


class FeedbackSummary(BaseModel):
    """
    Comprehensive feedback summary for reporting.

    Used in weekly/monthly feedback reports and dashboards.
    """

    period: str = Field(..., description="Report period (e.g., 'Week 46 2025')")
    stats: FeedbackStats = Field(..., description="Aggregated statistics")
    trends: List[SatisfactionTrend] = Field(default_factory=list, description="Time-series trends")

    # Top issues and improvements
    top_positive_comments: List[str] = Field(default_factory=list, description="Most helpful positive feedback")
    top_negative_comments: List[str] = Field(default_factory=list, description="Most critical issues")

    # Actionable insights
    improvement_opportunities: List[str] = Field(default_factory=list, description="Recommended improvements")
    underperforming_optimizations: List[str] = Field(default_factory=list, description="Low-rated optimization IDs")

    # Accuracy metrics
    high_accuracy_count: int = Field(0, description="Predictions within 10% of actual")
    low_accuracy_count: int = Field(0, description="Predictions off by >25%")

    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Report generation time")

    @validator('top_positive_comments', 'top_negative_comments')
    def limit_comment_count(cls, v: List[str]) -> List[str]:
        """Limit to top 5 comments."""
        return v[:5]


class FeedbackAlert(BaseModel):
    """
    Alert for critical feedback issues.

    Triggers notifications when user satisfaction drops or critical issues detected.
    """

    alert_id: str = Field(..., description="Unique alert identifier")
    severity: str = Field(..., description="critical|warning|info")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Detailed alert message")

    # Trigger conditions
    triggered_by: str = Field(..., description="What triggered the alert")
    threshold_value: float = Field(..., description="Threshold that was breached")
    actual_value: float = Field(..., description="Actual observed value")

    # Related data
    affected_optimizations: List[str] = Field(default_factory=list, description="Affected optimization IDs")
    related_feedback_ids: List[str] = Field(default_factory=list, description="Related feedback submissions")

    created_at: datetime = Field(default_factory=datetime.utcnow, description="Alert creation time")
    acknowledged: bool = Field(False, description="Alert acknowledgement status")

    @validator('severity')
    def validate_severity(cls, v: str) -> str:
        """Ensure valid severity level."""
        valid_levels = {'critical', 'warning', 'info'}
        if v.lower() not in valid_levels:
            raise ValueError(f"Severity must be one of {valid_levels}")
        return v.lower()
