"""
Explanation Data Models for GL-016 Waterguard

This module defines Pydantic schemas for all explainability-related data structures.
All explanations are derived from structured data - NO generative AI.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ExplanationMethod(str, Enum):
    """Supported explanation methods."""
    SHAP = "shap"
    LIME = "lime"
    SHAP_LIME_COMBINED = "shap_lime_combined"


class FeatureDirection(str, Enum):
    """Direction of feature effect on prediction."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    NEUTRAL = "neutral"


class RecommendationType(str, Enum):
    """Types of water treatment recommendations."""
    BLOWDOWN_INCREASE = "blowdown_increase"
    BLOWDOWN_DECREASE = "blowdown_decrease"
    DOSING_INCREASE = "dosing_increase"
    DOSING_DECREASE = "dosing_decrease"
    MAINTAIN_CURRENT = "maintain_current"
    EMERGENCY_ACTION = "emergency_action"


class FeatureContribution(BaseModel):
    """Single feature contribution to a prediction."""
    feature_name: str = Field(..., description="Name of the feature")
    value: float = Field(..., description="Current value of the feature")
    contribution: float = Field(..., description="Contribution to prediction")
    direction: FeatureDirection = Field(..., description="Effect direction")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    percentile: Optional[float] = Field(None, ge=0, le=100)


class LocalExplanation(BaseModel):
    """Local explanation for a single prediction/recommendation."""
    explanation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    recommendation_id: str = Field(..., description="ID of recommendation explained")
    method: ExplanationMethod = Field(..., description="Explanation method used")
    features: List[FeatureContribution] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)
    base_value: Optional[float] = Field(None)
    prediction_value: Optional[float] = Field(None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = Field(..., description="Model version used")
    is_reliable: bool = Field(True)
    warning_messages: List[str] = Field(default_factory=list)

    def get_top_features(self, n: int = 5) -> List[FeatureContribution]:
        """Get top N features by absolute contribution."""
        sorted_features = sorted(
            self.features,
            key=lambda x: abs(x.contribution),
            reverse=True
        )
        return sorted_features[:n]

    def get_positive_contributors(self) -> List[FeatureContribution]:
        """Get features with positive contributions."""
        return [f for f in self.features if f.contribution > 0]

    def get_negative_contributors(self) -> List[FeatureContribution]:
        """Get features with negative contributions."""
        return [f for f in self.features if f.contribution < 0]


class GlobalExplanation(BaseModel):
    """Global explanation summarizing model behavior across all predictions."""
    explanation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_version: str = Field(..., description="Model version explained")
    feature_importances: Dict[str, float] = Field(...)
    feature_importance_std: Dict[str, float] = Field(default_factory=dict)
    sample_size: int = Field(..., ge=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    computation_time_seconds: Optional[float] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_ranked_features(self) -> List[tuple]:
        """Get features ranked by importance."""
        return sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )


class ExplanationStabilityMetrics(BaseModel):
    """Metrics for assessing explanation stability and reliability."""
    explanation_id: str = Field(...)
    lipschitz_constant: Optional[float] = Field(None)
    perturbation_consistency: float = Field(..., ge=0, le=1)
    feature_agreement_score: Optional[float] = Field(None, ge=0, le=1)
    confidence_interval_width: Optional[float] = Field(None)
    is_out_of_distribution: bool = Field(False)
    mahalanobis_distance: Optional[float] = Field(None)
    passed_stability_check: bool = Field(True)
    stability_warnings: List[str] = Field(default_factory=list)


class ExplanationPayload(BaseModel):
    """Structured payload for UI consumption."""
    recommendation_id: str = Field(...)
    recommendation_type: RecommendationType = Field(...)
    recommendation_value: Optional[float] = Field(None)
    explanation_summary: str = Field(...)
    top_factors: List[Dict[str, Any]] = Field(...)
    confidence_level: str = Field(...)
    confidence_score: float = Field(..., ge=0, le=1)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    method_used: str = Field(...)
    warnings: List[str] = Field(default_factory=list)
    detailed_explanation: Optional[LocalExplanation] = Field(None)

    @classmethod
    def from_local_explanation(
        cls,
        explanation: LocalExplanation,
        recommendation_type: RecommendationType,
        recommendation_value: Optional[float],
        summary: str,
        top_n: int = 5
    ) -> "ExplanationPayload":
        """Create payload from a local explanation."""
        top_features = explanation.get_top_features(top_n)
        if explanation.confidence >= 0.8:
            confidence_level = "high"
        elif explanation.confidence >= 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        top_factors = [
            {
                "name": f.feature_name,
                "value": f.value,
                "unit": f.unit,
                "impact": "positive" if f.contribution > 0 else "negative",
                "contribution": abs(f.contribution),
                "direction": f.direction.value
            }
            for f in top_features
        ]
        return cls(
            recommendation_id=explanation.recommendation_id,
            recommendation_type=recommendation_type,
            recommendation_value=recommendation_value,
            explanation_summary=summary,
            top_factors=top_factors,
            confidence_level=confidence_level,
            confidence_score=explanation.confidence,
            method_used=explanation.method.value,
            warnings=explanation.warning_messages,
            detailed_explanation=explanation if explanation.is_reliable else None
        )
