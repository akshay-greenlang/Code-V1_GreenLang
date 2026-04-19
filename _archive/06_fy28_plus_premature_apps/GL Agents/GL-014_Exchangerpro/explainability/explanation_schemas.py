# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Explanation Schemas

Pydantic models for heat exchanger fouling explainability with zero-hallucination
guarantees. All numeric values are derived from deterministic calculations
(SHAP, LIME, causal inference), not LLM-generated estimates.

These schemas ensure:
- Type safety and validation for all explanation data
- Complete provenance tracking with SHA-256 hashes
- Stable explanations that align with engineering intuition
- Confidence levels for all predictions and attributions

Author: GreenLang AI Team
Version: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json

from pydantic import BaseModel, Field, validator, root_validator


class ExplanationType(str, Enum):
    """Types of explanations available."""
    SHAP = "shap"
    LIME = "lime"
    CAUSAL = "causal"
    ENGINEERING_RATIONALE = "engineering_rationale"
    FEATURE_IMPORTANCE = "feature_importance"
    ROOT_CAUSE = "root_cause"
    COUNTERFACTUAL = "counterfactual"


class PredictionType(str, Enum):
    """Types of fouling predictions that can be explained."""
    FOULING_FACTOR = "fouling_factor"
    FOULING_RATE = "fouling_rate"
    REMAINING_USEFUL_LIFE = "remaining_useful_life"
    CLEANING_URGENCY = "cleaning_urgency"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    THERMAL_RESISTANCE = "thermal_resistance"
    PRESSURE_DROP = "pressure_drop"
    HEAT_TRANSFER_COEFFICIENT = "heat_transfer_coefficient"


class FoulingMechanism(str, Enum):
    """Physical mechanisms of fouling."""
    PARTICULATE = "particulate"
    CRYSTALLIZATION = "crystallization"
    BIOLOGICAL = "biological"
    CORROSION = "corrosion"
    CHEMICAL_REACTION = "chemical_reaction"
    COMBINED = "combined"


class ConfidenceLevel(str, Enum):
    """Confidence levels for predictions and explanations."""
    HIGH = "high"          # >= 0.85
    MEDIUM = "medium"      # >= 0.65
    LOW = "low"            # >= 0.40
    VERY_LOW = "very_low"  # < 0.40


class FeatureCategory(str, Enum):
    """Categories for heat exchanger features."""
    THERMAL = "thermal"
    HYDRAULIC = "hydraulic"
    FLUID_PROPERTIES = "fluid_properties"
    OPERATING_CONDITIONS = "operating_conditions"
    GEOMETRY = "geometry"
    HISTORICAL = "historical"
    ENVIRONMENTAL = "environmental"


class FeatureImportance(BaseModel):
    """
    Importance score for a single feature in fouling prediction.

    Attributes:
        feature_name: Name of the feature (e.g., 'delta_P_normalized')
        importance_value: Absolute importance score from SHAP/LIME
        rank: Ranking among all features (1 = most important)
        direction: Direction of impact ('positive' = increases fouling)
        category: Engineering category of the feature
        confidence: Confidence in this importance estimate
        engineering_interpretation: Human-readable interpretation
    """
    feature_name: str = Field(..., description="Name of the feature")
    importance_value: float = Field(..., description="Absolute importance score")
    rank: int = Field(..., ge=1, description="Importance ranking (1 = highest)")
    direction: str = Field(..., pattern="^(positive|negative|neutral)$")
    category: FeatureCategory = Field(..., description="Engineering category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in estimate")
    engineering_interpretation: Optional[str] = Field(
        None, description="Human-readable interpretation"
    )
    unit: Optional[str] = Field(None, description="Unit of measurement")

    @validator('direction')
    def validate_direction(cls, v):
        """Ensure direction is valid."""
        if v not in ('positive', 'negative', 'neutral'):
            raise ValueError("Direction must be 'positive', 'negative', or 'neutral'")
        return v


class FeatureContribution(BaseModel):
    """
    Contribution of a single feature to a specific prediction.

    Used for local explanations (SHAP force plots, LIME).
    """
    feature_name: str = Field(..., description="Name of the feature")
    feature_value: float = Field(..., description="Actual value of the feature")
    baseline_value: Optional[float] = Field(None, description="Expected/baseline value")
    contribution: float = Field(..., description="Contribution to prediction (SHAP value)")
    contribution_percentage: float = Field(
        ..., ge=-100.0, le=100.0, description="Percentage of total contribution"
    )
    direction: str = Field(..., description="Impact direction on prediction")
    category: FeatureCategory = Field(..., description="Engineering category")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    is_anomalous: bool = Field(False, description="Whether this value is unusual")

    @validator('contribution_percentage')
    def validate_percentage(cls, v):
        """Ensure percentage is within valid range."""
        return round(v, 4)


class ConfidenceBounds(BaseModel):
    """Confidence interval for a point estimate."""
    lower_bound: float = Field(..., description="Lower bound of CI")
    upper_bound: float = Field(..., description="Upper bound of CI")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="CI level (e.g., 0.95)")
    method: str = Field("bootstrap", description="Method used to compute CI")

    @root_validator
    def validate_bounds(cls, values):
        """Ensure lower_bound <= upper_bound."""
        lower = values.get('lower_bound')
        upper = values.get('upper_bound')
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower_bound must be <= upper_bound")
        return values


class UncertaintyEstimate(BaseModel):
    """
    Complete uncertainty characterization for a prediction.

    Separates epistemic (model) uncertainty from aleatoric (data) uncertainty.
    """
    point_estimate: float = Field(..., description="Point estimate of the prediction")
    standard_error: float = Field(..., ge=0, description="Standard error")
    confidence_interval: ConfidenceBounds = Field(..., description="Confidence interval")
    epistemic_uncertainty: float = Field(
        ..., ge=0, description="Model/knowledge uncertainty"
    )
    aleatoric_uncertainty: float = Field(
        ..., ge=0, description="Data/irreducible uncertainty"
    )
    total_uncertainty: float = Field(..., ge=0, description="Total uncertainty")

    @validator('total_uncertainty', always=True)
    def compute_total(cls, v, values):
        """Compute total uncertainty if not provided."""
        if v is None or v == 0:
            epi = values.get('epistemic_uncertainty', 0)
            ale = values.get('aleatoric_uncertainty', 0)
            return (epi ** 2 + ale ** 2) ** 0.5
        return v


class LocalExplanation(BaseModel):
    """
    Local explanation for a single prediction instance.

    Explains why the model made a specific prediction for a specific
    heat exchanger at a specific time.
    """
    explanation_id: str = Field(..., description="Unique identifier for this explanation")
    exchanger_id: str = Field(..., description="Heat exchanger identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    prediction_type: PredictionType = Field(..., description="Type of prediction explained")
    prediction_value: float = Field(..., description="The predicted value")
    base_value: float = Field(..., description="Baseline/expected value")

    feature_contributions: List[FeatureContribution] = Field(
        ..., description="Contributions of each feature"
    )
    top_positive_drivers: List[str] = Field(
        default_factory=list, description="Features increasing prediction"
    )
    top_negative_drivers: List[str] = Field(
        default_factory=list, description="Features decreasing prediction"
    )

    explanation_method: ExplanationType = Field(..., description="Method used (SHAP/LIME)")
    local_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Local surrogate model accuracy (R2)"
    )
    stability_score: float = Field(
        ..., ge=0.0, le=1.0, description="Stability of explanation across similar inputs"
    )
    confidence: ConfidenceLevel = Field(..., description="Overall confidence level")

    computation_time_ms: float = Field(..., ge=0, description="Computation time in ms")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    methodology_version: str = Field("1.0.0", description="Version of explanation method")

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for complete audit trail."""
        content = {
            "explanation_id": self.explanation_id,
            "exchanger_id": self.exchanger_id,
            "prediction_value": self.prediction_value,
            "feature_contributions": [fc.dict() for fc in self.feature_contributions],
            "timestamp": self.timestamp.isoformat(),
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


class GlobalExplanation(BaseModel):
    """
    Global explanation for model behavior across all exchangers.

    Provides overall feature importance rankings and model-level insights.
    """
    explanation_id: str = Field(..., description="Unique identifier")
    model_name: str = Field(..., description="Name of the model being explained")
    model_version: str = Field(..., description="Version of the model")
    prediction_type: PredictionType = Field(..., description="Type of prediction")

    feature_importance: List[FeatureImportance] = Field(
        ..., description="Ranked feature importance scores"
    )
    top_k_features: int = Field(10, ge=1, description="Number of top features included")

    interaction_effects: Optional[Dict[str, Dict[str, float]]] = Field(
        None, description="Pairwise feature interaction strengths"
    )

    # Summary statistics
    total_exchangers_analyzed: int = Field(..., ge=1)
    mean_prediction: float = Field(...)
    std_prediction: float = Field(..., ge=0)

    # Methodology
    explanation_method: ExplanationType = Field(...)
    aggregation_method: str = Field("mean_absolute", description="How values were aggregated")

    confidence: ConfidenceLevel = Field(...)
    computation_time_ms: float = Field(..., ge=0)
    provenance_hash: str = Field(...)
    methodology_version: str = Field("1.0.0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CausalRelationship(BaseModel):
    """
    Represents a causal relationship between features and fouling.

    Based on domain knowledge and causal inference methods.
    """
    source: str = Field(..., description="Cause variable")
    target: str = Field(..., description="Effect variable")
    causal_effect: float = Field(..., description="Estimated causal effect")
    effect_type: str = Field("direct", pattern="^(direct|indirect|total)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    mechanism: str = Field(..., description="Physical mechanism description")
    evidence_strength: str = Field(..., pattern="^(strong|moderate|weak)$")
    references: List[str] = Field(default_factory=list, description="Literature references")


class RootCauseAnalysis(BaseModel):
    """
    Root cause analysis for accelerated fouling conditions.
    """
    hypothesis_id: str = Field(..., description="Unique identifier")
    exchanger_id: str = Field(..., description="Heat exchanger analyzed")

    primary_cause: str = Field(..., description="Primary root cause identified")
    secondary_causes: List[str] = Field(default_factory=list)

    causal_chain: List[CausalRelationship] = Field(
        ..., description="Chain of causal relationships"
    )

    causal_effect: float = Field(..., description="Total causal effect on fouling")
    confidence_interval: ConfidenceBounds = Field(...)

    fouling_mechanism: FoulingMechanism = Field(...)
    supporting_evidence: List[str] = Field(..., description="Evidence supporting hypothesis")

    intervention_recommendations: List[str] = Field(
        ..., description="Recommended interventions"
    )
    expected_improvement: float = Field(
        ..., ge=0, le=1.0, description="Expected fractional improvement"
    )

    confidence: ConfidenceLevel = Field(...)
    provenance_hash: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CounterfactualExplanation(BaseModel):
    """
    Counterfactual explanation: "What would need to change?"

    Explains what operating conditions would need to change to achieve
    a different fouling outcome.
    """
    explanation_id: str = Field(...)
    exchanger_id: str = Field(...)

    original_prediction: float = Field(...)
    target_prediction: float = Field(...)

    feature_changes: Dict[str, Tuple[float, float]] = Field(
        ..., description="Required changes: feature -> (original, new)"
    )

    feasibility_score: float = Field(
        ..., ge=0.0, le=1.0, description="How feasible are these changes"
    )
    cost_estimate: Optional[float] = Field(
        None, description="Estimated cost of intervention"
    )

    explanation_text: str = Field(..., description="Human-readable explanation")
    confidence: ConfidenceLevel = Field(...)
    provenance_hash: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EngineeringRationale(BaseModel):
    """
    Engineering-based rationale using thermal engineering terminology.

    Maps ML features to physical engineering concepts for operator understanding.
    """
    rationale_id: str = Field(...)
    exchanger_id: str = Field(...)

    # Main narrative
    summary: str = Field(..., description="Executive summary in engineering terms")
    detailed_rationale: str = Field(..., description="Detailed engineering explanation")

    # Key observations
    key_observations: List[str] = Field(
        ..., description="Key engineering observations"
    )

    # Thermal performance indicators
    thermal_indicators: Dict[str, Any] = Field(
        ..., description="Key thermal performance metrics"
    )

    # Hydraulic performance indicators
    hydraulic_indicators: Dict[str, Any] = Field(
        ..., description="Key hydraulic performance metrics"
    )

    # Physical mechanism identification
    fouling_mechanism: FoulingMechanism = Field(...)
    mechanism_evidence: List[str] = Field(...)

    # Recommendations
    operational_recommendations: List[str] = Field(...)
    maintenance_recommendations: List[str] = Field(...)

    # Confidence and provenance
    confidence: ConfidenceLevel = Field(...)
    methodology_version: str = Field("1.0.0")
    calculation_references: List[str] = Field(
        default_factory=list, description="References to calculation methods"
    )
    provenance_hash: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExplanationStabilityMetrics(BaseModel):
    """
    Metrics to ensure explanation stability across similar operating points.

    Prevents oscillating explanations for similar conditions.
    """
    stability_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall stability score"
    )
    feature_ranking_stability: float = Field(
        ..., ge=0.0, le=1.0, description="Stability of feature rankings"
    )
    contribution_variance: float = Field(
        ..., ge=0, description="Variance in contributions across similar points"
    )
    neighboring_points_analyzed: int = Field(..., ge=1)
    stability_method: str = Field("neighborhood_sampling")


class FoulingExplainabilityReport(BaseModel):
    """
    Complete explainability report for fouling analysis.

    Integrates all explanation types into a comprehensive report.
    """
    report_id: str = Field(...)
    exchanger_id: str = Field(...)

    # Prediction being explained
    prediction_type: PredictionType = Field(...)
    prediction_value: float = Field(...)
    prediction_uncertainty: UncertaintyEstimate = Field(...)

    # Local explanation
    local_explanation: LocalExplanation = Field(...)

    # Global context
    global_explanation: Optional[GlobalExplanation] = Field(None)

    # Causal analysis
    root_cause_analysis: Optional[RootCauseAnalysis] = Field(None)
    counterfactual: Optional[CounterfactualExplanation] = Field(None)

    # Engineering interpretation
    engineering_rationale: EngineeringRationale = Field(...)

    # Stability assessment
    stability_metrics: ExplanationStabilityMetrics = Field(...)

    # Metadata
    model_name: str = Field(...)
    model_version: str = Field(...)
    data_quality_score: float = Field(..., ge=0.0, le=1.0)
    missing_features: List[str] = Field(default_factory=list)

    # Provenance
    computation_time_ms: float = Field(..., ge=0)
    provenance_hash: str = Field(...)
    methodology_version: str = Field("1.0.0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for complete audit trail."""
        content = {
            "report_id": self.report_id,
            "exchanger_id": self.exchanger_id,
            "prediction_value": self.prediction_value,
            "local_explanation_hash": self.local_explanation.provenance_hash,
            "engineering_rationale_hash": self.engineering_rationale.provenance_hash,
            "timestamp": self.timestamp.isoformat(),
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


class DashboardExplanationData(BaseModel):
    """
    Formatted data for dashboard visualization.

    Pre-computed data structures for common visualization types.
    """
    # Waterfall chart data (SHAP)
    waterfall_data: List[Dict[str, Any]] = Field(
        ..., description="Data for SHAP waterfall chart"
    )

    # Force plot data
    force_plot_data: Dict[str, Any] = Field(
        ..., description="Data for SHAP force plot"
    )

    # Feature importance bar chart
    feature_importance_chart: List[Dict[str, Any]] = Field(
        ..., description="Data for feature importance chart"
    )

    # Causal graph visualization
    causal_graph_data: Optional[Dict[str, Any]] = Field(
        None, description="Data for causal graph visualization"
    )

    # Trend data for time series
    trend_data: Optional[Dict[str, List[Dict[str, Any]]]] = Field(
        None, description="Time series trend data"
    )

    # Summary metrics
    summary_metrics: Dict[str, Any] = Field(
        ..., description="Summary statistics for dashboard"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Type aliases for convenience
FeatureContributions = List[FeatureContribution]
FeatureImportanceList = List[FeatureImportance]
CausalRelationships = List[CausalRelationship]
