# -*- coding: utf-8 -*-
"""
Explanation Schemas for GL-001 ThermalCommand Explainability Module.

Defines Pydantic models for ML explanations with zero-hallucination guarantees.
All numeric values in explanations are derived from deterministic calculations,
not LLM-generated estimates.

Author: GreenLang AI Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import hashlib


class ExplanationType(str, Enum):
    """Types of explanations supported."""

    SHAP = "shap"
    LIME = "lime"
    COUNTERFACTUAL = "counterfactual"
    FEATURE_IMPORTANCE = "feature_importance"
    DECISION_PATH = "decision_path"


class PredictionType(str, Enum):
    """Types of predictions that can be explained."""

    DEMAND_FORECAST = "demand_forecast"
    HEALTH_SCORE = "health_score"
    ANOMALY_DETECTION = "anomaly_detection"
    OPTIMIZATION_DECISION = "optimization_decision"
    EFFICIENCY_PREDICTION = "efficiency_prediction"
    EMISSIONS_PREDICTION = "emissions_prediction"


class ConfidenceLevel(str, Enum):
    """Confidence levels for explanations."""

    HIGH = "high"  # >= 90%
    MEDIUM = "medium"  # >= 70% and < 90%
    LOW = "low"  # < 70%


class FeatureContribution(BaseModel):
    """
    Represents a single feature's contribution to a prediction.

    Zero-hallucination: All values are computed deterministically
    from SHAP/LIME algorithms, not LLM-generated.
    """

    feature_name: str = Field(..., description="Name of the feature")
    feature_value: float = Field(..., description="Actual value of the feature")
    contribution: float = Field(..., description="Contribution to prediction (positive or negative)")
    contribution_percentage: float = Field(..., description="Percentage contribution to total")
    direction: str = Field(..., description="Direction of impact: 'positive' or 'negative'")
    unit: Optional[str] = Field(None, description="Unit of measurement for the feature")
    baseline_value: Optional[float] = Field(None, description="Baseline/expected value for comparison")

    @validator('direction')
    def validate_direction(cls, v):
        """Ensure direction is valid."""
        if v not in ['positive', 'negative', 'neutral']:
            raise ValueError("Direction must be 'positive', 'negative', or 'neutral'")
        return v

    @validator('contribution_percentage')
    def validate_percentage(cls, v):
        """Ensure percentage is valid."""
        if v < -100 or v > 100:
            raise ValueError("Contribution percentage must be between -100 and 100")
        return v


class ConfidenceBounds(BaseModel):
    """
    Confidence bounds for predictions and explanations.

    Provides uncertainty quantification for transparent decision-making.
    """

    lower_bound: float = Field(..., description="Lower bound of confidence interval")
    upper_bound: float = Field(..., description="Upper bound of confidence interval")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="Confidence level (e.g., 0.95 for 95%)")
    method: str = Field("bootstrap", description="Method used to compute bounds")

    @validator('upper_bound')
    def validate_bounds(cls, v, values):
        """Ensure upper bound is greater than lower bound."""
        if 'lower_bound' in values and v < values['lower_bound']:
            raise ValueError("Upper bound must be >= lower bound")
        return v


class UncertaintyRange(BaseModel):
    """
    Uncertainty quantification for model predictions.

    Essential for regulatory compliance and audit trails.
    """

    point_estimate: float = Field(..., description="Point estimate of the prediction")
    standard_error: float = Field(..., ge=0, description="Standard error of the estimate")
    confidence_interval: ConfidenceBounds = Field(..., description="Confidence interval")
    prediction_variance: float = Field(..., ge=0, description="Variance of prediction")
    epistemic_uncertainty: float = Field(..., ge=0, description="Model uncertainty (reducible)")
    aleatoric_uncertainty: float = Field(..., ge=0, description="Data uncertainty (irreducible)")

    @property
    def total_uncertainty(self) -> float:
        """Calculate total uncertainty."""
        return self.epistemic_uncertainty + self.aleatoric_uncertainty


class Counterfactual(BaseModel):
    """
    Counterfactual explanation: what would need to change for a different outcome.

    Answers questions like: "What would change if temperature increased by 10C?"
    """

    counterfactual_id: str = Field(..., description="Unique identifier for this counterfactual")
    original_prediction: float = Field(..., description="Original prediction value")
    target_prediction: float = Field(..., description="Target prediction value")
    feature_changes: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Features that need to change: {feature: {from: x, to: y}}"
    )
    feasibility_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="How feasible is this counterfactual (0-1)"
    )
    sparsity: int = Field(..., ge=1, description="Number of features that need to change")
    distance: float = Field(..., ge=0, description="Distance from original instance")
    validity: bool = Field(..., description="Whether counterfactual satisfies constraints")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When generated")

    @property
    def is_actionable(self) -> bool:
        """Check if counterfactual is actionable (feasible and valid)."""
        return self.feasibility_score > 0.5 and self.validity


class SHAPExplanation(BaseModel):
    """
    SHAP-based explanation for a prediction.

    Provides game-theoretic feature attribution.
    """

    explanation_id: str = Field(..., description="Unique explanation identifier")
    prediction_type: PredictionType = Field(..., description="Type of prediction explained")
    base_value: float = Field(..., description="Expected model output (average prediction)")
    prediction_value: float = Field(..., description="Actual prediction for this instance")
    feature_contributions: List[FeatureContribution] = Field(
        ...,
        description="List of feature contributions"
    )
    interaction_effects: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Feature interaction effects"
    )
    consistency_check: float = Field(
        ...,
        ge=0,
        le=0.01,
        description="Sum of SHAP values should equal prediction - base_value (tolerance)"
    )
    explainer_type: str = Field(..., description="SHAP explainer type: tree, kernel, deep")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When generated")
    computation_time_ms: float = Field(..., ge=0, description="Time to compute explanation")
    random_seed: int = Field(42, description="Random seed for reproducibility")

    @validator('feature_contributions')
    def validate_contributions(cls, v):
        """Ensure at least one contribution exists."""
        if not v:
            raise ValueError("At least one feature contribution required")
        return v


class LIMEExplanation(BaseModel):
    """
    LIME-based explanation for a prediction.

    Provides local surrogate model explanations.
    """

    explanation_id: str = Field(..., description="Unique explanation identifier")
    prediction_type: PredictionType = Field(..., description="Type of prediction explained")
    prediction_value: float = Field(..., description="Prediction for this instance")
    feature_contributions: List[FeatureContribution] = Field(
        ...,
        description="Feature contributions from local model"
    )
    local_model_r2: float = Field(
        ...,
        ge=0,
        le=1,
        description="R-squared of local surrogate model"
    )
    local_model_intercept: float = Field(..., description="Intercept of local linear model")
    neighborhood_size: int = Field(..., ge=100, description="Number of samples in neighborhood")
    kernel_width: float = Field(..., gt=0, description="Kernel width for weighting")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When generated")
    computation_time_ms: float = Field(..., ge=0, description="Time to compute explanation")
    random_seed: int = Field(42, description="Random seed for reproducibility")

    @property
    def is_reliable(self) -> bool:
        """Check if LIME explanation is reliable (high local fidelity)."""
        return self.local_model_r2 >= 0.8


class DecisionExplanation(BaseModel):
    """
    Explanation for optimization decisions.

    Explains why the optimizer chose a particular configuration.
    """

    decision_id: str = Field(..., description="Unique decision identifier")
    objective_value: float = Field(..., description="Objective function value achieved")
    binding_constraints: List[str] = Field(
        ...,
        description="Constraints that are active/binding"
    )
    shadow_prices: Dict[str, float] = Field(
        ...,
        description="Shadow prices for constraints (marginal value)"
    )
    reduced_costs: Dict[str, float] = Field(
        ...,
        description="Reduced costs for variables"
    )
    sensitivity_analysis: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Sensitivity ranges for coefficients"
    )
    alternative_solutions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative near-optimal solutions"
    )
    optimality_gap: float = Field(
        ...,
        ge=0,
        le=1,
        description="Gap from optimal solution (for MIP)"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When generated")


class ExplanationReport(BaseModel):
    """
    Complete explanation report for a prediction or decision.

    Combines multiple explanation methods for comprehensive transparency.
    """

    report_id: str = Field(..., description="Unique report identifier")
    prediction_type: PredictionType = Field(..., description="Type of prediction explained")
    model_name: str = Field(..., description="Name of the model being explained")
    model_version: str = Field(..., description="Version of the model")

    # Input data
    input_features: Dict[str, float] = Field(..., description="Input features used for prediction")

    # Prediction and uncertainty
    prediction_value: float = Field(..., description="Model prediction")
    uncertainty: UncertaintyRange = Field(..., description="Prediction uncertainty")
    confidence_level: ConfidenceLevel = Field(..., description="Overall confidence in explanation")

    # Explanations
    shap_explanation: Optional[SHAPExplanation] = Field(None, description="SHAP explanation")
    lime_explanation: Optional[LIMEExplanation] = Field(None, description="LIME explanation")
    decision_explanation: Optional[DecisionExplanation] = Field(None, description="Decision explanation")
    counterfactuals: List[Counterfactual] = Field(
        default_factory=list,
        description="Counterfactual explanations"
    )

    # Summary
    top_features: List[FeatureContribution] = Field(
        ...,
        description="Top contributing features (sorted by importance)"
    )
    narrative_summary: str = Field(
        ...,
        description="Human-readable summary (for display, not for calculations)"
    )

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Report generation time")
    computation_time_ms: float = Field(..., ge=0, description="Total computation time")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    deterministic: bool = Field(True, description="Whether all values are deterministically computed")

    @validator('top_features')
    def validate_top_features(cls, v):
        """Ensure features are sorted by absolute contribution."""
        return sorted(v, key=lambda x: abs(x.contribution), reverse=True)

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for audit trail."""
        content = f"{self.report_id}{self.prediction_value}{self.timestamp}{self.input_features}"
        return hashlib.sha256(content.encode()).hexdigest()

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchExplanationSummary(BaseModel):
    """
    Summary of explanations for a batch of predictions.

    Useful for dashboard displays and aggregate analysis.
    """

    summary_id: str = Field(..., description="Unique summary identifier")
    prediction_type: PredictionType = Field(..., description="Type of predictions explained")
    batch_size: int = Field(..., ge=1, description="Number of predictions in batch")

    # Aggregate feature importance
    global_feature_importance: Dict[str, float] = Field(
        ...,
        description="Aggregate feature importance across batch"
    )
    feature_importance_std: Dict[str, float] = Field(
        ...,
        description="Standard deviation of feature importance"
    )

    # Statistics
    mean_prediction: float = Field(..., description="Mean prediction value")
    std_prediction: float = Field(..., ge=0, description="Standard deviation of predictions")
    mean_confidence: float = Field(..., ge=0, le=1, description="Mean confidence level")

    # Explanation quality metrics
    mean_shap_consistency: float = Field(..., ge=0, le=0.1, description="Mean SHAP consistency error")
    mean_lime_r2: float = Field(..., ge=0, le=1, description="Mean LIME local R-squared")

    # Common patterns
    common_binding_constraints: List[str] = Field(
        default_factory=list,
        description="Most common binding constraints"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Summary generation time")
    computation_time_ms: float = Field(..., ge=0, description="Total computation time for batch")


class DashboardExplanationData(BaseModel):
    """
    Explanation data formatted for dashboard visualization.

    Provides pre-computed data for charts and graphs.
    """

    # Waterfall chart data
    waterfall_data: List[Dict[str, Any]] = Field(
        ...,
        description="Data for waterfall chart visualization"
    )

    # Force plot data
    force_plot_data: Dict[str, Any] = Field(
        ...,
        description="Data for SHAP force plot"
    )

    # Feature importance bar chart
    feature_importance_chart: List[Dict[str, Any]] = Field(
        ...,
        description="Data for feature importance bar chart"
    )

    # Beeswarm/summary plot data
    summary_plot_data: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Data for SHAP summary plot"
    )

    # Time series of feature importance
    importance_over_time: Optional[Dict[str, List[Dict[str, Any]]]] = Field(
        None,
        description="Feature importance trends over time"
    )

    # Counterfactual visualization data
    counterfactual_chart: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Data for counterfactual comparison chart"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Data generation time")


# Type aliases for convenience
FeatureContributions = List[FeatureContribution]
CounterfactualList = List[Counterfactual]
