"""
Pydantic schemas for ML Explainability Framework.

This module defines all data models used throughout the explainability
framework, ensuring type safety, validation, and serialization support.

Example:
    >>> result = ExplanationResult(
    ...     prediction=0.85,
    ...     feature_contributions={"temperature": 0.35, "pressure": 0.25},
    ...     top_features=[("temperature", 0.35), ("pressure", 0.25)],
    ...     confidence=0.92,
    ...     explanation_text="High fouling risk due to elevated temperature",
    ...     provenance_hash="abc123..."
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import hashlib
import json

from pydantic import BaseModel, Field, validator


class ExplainerType(str, Enum):
    """Supported explainer types."""

    SHAP_TREE = "shap_tree"
    SHAP_KERNEL = "shap_kernel"
    SHAP_LINEAR = "shap_linear"
    SHAP_DEEP = "shap_deep"
    LIME_TABULAR = "lime_tabular"
    COUNTERFACTUAL = "counterfactual"
    HUMAN_READABLE = "human_readable"


class ModelType(str, Enum):
    """Supported model types for explanation."""

    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    GENERIC = "generic"


class ExplanationLevel(str, Enum):
    """Level of explanation detail."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    TECHNICAL = "technical"


class FeatureContribution(BaseModel):
    """Individual feature contribution to a prediction."""

    feature_name: str = Field(..., description="Name of the feature")
    value: float = Field(..., description="Actual feature value")
    contribution: float = Field(..., description="Contribution to prediction (SHAP value)")
    direction: str = Field(..., description="Direction of impact: 'positive' or 'negative'")
    percentage: float = Field(..., ge=0, le=100, description="Percentage contribution")

    @validator("direction")
    def validate_direction(cls, v: str) -> str:
        """Validate direction is either positive or negative."""
        if v not in ("positive", "negative"):
            raise ValueError("Direction must be 'positive' or 'negative'")
        return v


class ExplanationResult(BaseModel):
    """
    Complete explanation result for a single prediction.

    This is the primary output schema for all explainability methods,
    providing both numerical explanations and human-readable text.

    Attributes:
        prediction: The model's predicted value
        feature_contributions: Mapping of feature names to their contributions
        top_features: Ordered list of (feature_name, contribution) tuples
        confidence: Confidence score for the explanation (0-1)
        explanation_text: Human-readable explanation text
        provenance_hash: SHA-256 hash for audit trail
    """

    prediction: float = Field(..., description="Model prediction value")
    prediction_class: Optional[str] = Field(None, description="Classification label if applicable")
    feature_contributions: Dict[str, float] = Field(
        ...,
        description="Mapping of feature names to contribution values"
    )
    top_features: List[Tuple[str, float]] = Field(
        ...,
        description="Top contributing features ordered by absolute contribution"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the explanation"
    )
    explanation_text: Optional[str] = Field(
        None,
        description="Human-readable explanation"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for complete audit trail"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When explanation was generated"
    )
    explainer_type: ExplainerType = Field(
        ...,
        description="Type of explainer used"
    )
    model_type: ModelType = Field(
        default=ModelType.GENERIC,
        description="Type of model explained"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.dict()


class GlobalExplanationResult(BaseModel):
    """
    Global model explanation result.

    Provides overall feature importance and model-level insights
    rather than instance-specific explanations.
    """

    model_id: str = Field(..., description="Unique identifier for the model")
    feature_importance: Dict[str, float] = Field(
        ...,
        description="Global feature importance scores"
    )
    feature_rankings: List[Tuple[str, float]] = Field(
        ...,
        description="Features ranked by importance"
    )
    interaction_effects: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Feature interaction effects"
    )
    summary_statistics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics for explanations"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When explanation was generated"
    )
    sample_size: int = Field(..., ge=1, description="Number of samples used")
    explainer_type: ExplainerType = Field(..., description="Explainer type used")


class CounterfactualResult(BaseModel):
    """
    Counterfactual explanation result.

    Shows the minimal changes needed to alter a prediction,
    helping users understand decision boundaries.
    """

    original_prediction: float = Field(..., description="Original model prediction")
    original_class: Optional[str] = Field(None, description="Original class label")
    counterfactual_prediction: float = Field(..., description="Counterfactual prediction")
    counterfactual_class: Optional[str] = Field(None, description="Counterfactual class label")
    changes_required: Dict[str, Tuple[float, float]] = Field(
        ...,
        description="Feature changes: {feature: (original, new)}"
    )
    change_magnitude: float = Field(
        ...,
        ge=0,
        description="Total magnitude of changes"
    )
    num_features_changed: int = Field(
        ...,
        ge=0,
        description="Number of features modified"
    )
    feasibility_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="How feasible the counterfactual is"
    )
    explanation_text: Optional[str] = Field(
        None,
        description="Human-readable explanation"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WhatIfResult(BaseModel):
    """
    What-if scenario analysis result.

    Shows how prediction changes when specific features are modified.
    """

    scenario_name: str = Field(..., description="Name of the scenario")
    original_prediction: float = Field(..., description="Original prediction")
    modified_prediction: float = Field(..., description="Prediction after changes")
    prediction_change: float = Field(..., description="Change in prediction")
    feature_changes: Dict[str, Tuple[float, float]] = Field(
        ...,
        description="Applied changes: {feature: (old, new)}"
    )
    sensitivity: Dict[str, float] = Field(
        default_factory=dict,
        description="Sensitivity of prediction to each changed feature"
    )
    explanation_text: Optional[str] = Field(None, description="Human-readable explanation")
    provenance_hash: str = Field(..., description="SHA-256 hash")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExplanationRequest(BaseModel):
    """
    Request schema for explanation API endpoints.
    """

    model_id: str = Field(..., description="ID of the model to explain")
    instance_data: Dict[str, float] = Field(
        ...,
        description="Feature values for the instance"
    )
    explanation_level: ExplanationLevel = Field(
        default=ExplanationLevel.STANDARD,
        description="Level of detail"
    )
    include_counterfactual: bool = Field(
        default=False,
        description="Include counterfactual explanation"
    )
    include_human_readable: bool = Field(
        default=True,
        description="Include human-readable text"
    )
    top_k_features: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top features to return"
    )
    target_class: Optional[str] = Field(
        None,
        description="Target class for counterfactual"
    )


class CounterfactualRequest(BaseModel):
    """
    Request schema for counterfactual explanation endpoint.
    """

    model_id: str = Field(..., description="ID of the model")
    instance_data: Dict[str, float] = Field(..., description="Original instance")
    target_prediction: Optional[float] = Field(
        None,
        description="Target prediction value"
    )
    target_class: Optional[str] = Field(
        None,
        description="Target class for classification"
    )
    feature_constraints: Optional[Dict[str, Tuple[float, float]]] = Field(
        None,
        description="Constraints: {feature: (min, max)}"
    )
    immutable_features: Optional[List[str]] = Field(
        None,
        description="Features that cannot be changed"
    )
    max_features_to_change: int = Field(
        default=5,
        ge=1,
        description="Maximum features to modify"
    )


class GlobalExplanationRequest(BaseModel):
    """
    Request schema for global model explanation endpoint.
    """

    model_id: str = Field(..., description="ID of the model")
    sample_data: Optional[List[Dict[str, float]]] = Field(
        None,
        description="Sample data for explanation"
    )
    sample_size: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Number of samples to use"
    )
    include_interactions: bool = Field(
        default=False,
        description="Include feature interactions"
    )


class ProcessHeatContext(BaseModel):
    """
    Process heat domain-specific context for explanations.

    Provides domain knowledge to enhance explanation quality.
    """

    equipment_type: str = Field(..., description="Type of equipment (boiler, furnace, etc.)")
    process_type: str = Field(..., description="Process type (steam, combustion, etc.)")
    operating_mode: str = Field(default="normal", description="Operating mode")
    risk_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Domain-specific risk thresholds"
    )
    feature_units: Dict[str, str] = Field(
        default_factory=dict,
        description="Units for each feature"
    )
    feature_descriptions: Dict[str, str] = Field(
        default_factory=dict,
        description="Human-readable feature descriptions"
    )
    domain_rules: List[str] = Field(
        default_factory=list,
        description="Domain-specific rules to apply"
    )


def compute_provenance_hash(data: Union[Dict, BaseModel, str]) -> str:
    """
    Compute SHA-256 provenance hash for any data.

    Args:
        data: Input data (dict, Pydantic model, or string)

    Returns:
        SHA-256 hexdigest string
    """
    if isinstance(data, BaseModel):
        data_str = data.json(sort_keys=True)
    elif isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True, default=str)
    else:
        data_str = str(data)

    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()
