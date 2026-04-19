"""
Explainability Payload Models - Pydantic schemas for BURNMASTER explanations.

This module defines all data models for the explainability system, ensuring
type safety, validation, and serialization for both operator (plain language)
and engineer (technical detail) audiences.

Example:
    >>> from explainability_payload import PhysicsExplanation
    >>> explanation = PhysicsExplanation(
    ...     explanation_type="stoichiometry",
    ...     summary="Operating with 15% excess air",
    ...     confidence=0.95
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class ExplanationType(str, Enum):
    """Types of explanations available in the system."""

    PHYSICS = "physics"
    SHAP = "shap"
    LIME = "lime"
    CONSTRAINT = "constraint"
    RECOMMENDATION = "recommendation"
    COMPREHENSIVE = "comprehensive"


class AudienceLevel(str, Enum):
    """Target audience for explanation formatting."""

    OPERATOR = "operator"
    ENGINEER = "engineer"
    EXECUTIVE = "executive"


class ConfidenceLevel(str, Enum):
    """Qualitative confidence levels for explanations."""

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class ConstraintStatus(str, Enum):
    """Status of optimization constraints."""

    BINDING = "binding"
    ACTIVE = "active"
    INACTIVE = "inactive"
    VIOLATED = "violated"


class ImpactDirection(str, Enum):
    """Direction of predicted impact."""

    INCREASE = "increase"
    DECREASE = "decrease"
    NO_CHANGE = "no_change"
    UNCERTAIN = "uncertain"


# -----------------------------------------------------------------------------
# Base Models
# -----------------------------------------------------------------------------

class BaseExplanation(BaseModel):
    """Base class for all explanation types."""

    explanation_id: str = Field(..., description="Unique identifier for this explanation")
    explanation_type: ExplanationType = Field(..., description="Type of explanation")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When generated")
    summary: str = Field(..., description="Plain language summary")
    technical_detail: Optional[str] = Field(None, description="Technical explanation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    confidence_level: ConfidenceLevel = Field(..., description="Qualitative confidence level")
    uncertainty_bounds: Optional[Dict[str, float]] = Field(None, description="Uncertainty bounds")
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash for audit trail")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator("confidence_level", pre=True, always=True)
    def set_confidence_level(cls, v, values):
        """Automatically set confidence level from confidence score."""
        if v is not None:
            return v
        confidence = values.get("confidence", 0.5)
        if confidence > 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif confidence > 0.85:
            return ConfidenceLevel.HIGH
        elif confidence > 0.70:
            return ConfidenceLevel.MEDIUM
        elif confidence > 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class FeatureContribution(BaseModel):
    """Contribution of a single feature to a prediction."""

    feature_name: str = Field(..., description="Name of the feature")
    feature_value: float = Field(..., description="Current value of the feature")
    contribution: float = Field(..., description="Contribution to prediction")
    contribution_percent: float = Field(..., description="Percentage contribution")
    direction: ImpactDirection = Field(..., description="Direction of impact")
    unit: Optional[str] = Field(None, description="Engineering unit")
    description: str = Field(..., description="Plain language description")


class UncertaintyBounds(BaseModel):
    """Uncertainty bounds for a prediction or explanation."""

    lower_bound: float = Field(..., description="Lower bound")
    upper_bound: float = Field(..., description="Upper bound")
    confidence_interval: float = Field(0.90, description="Confidence interval")
    distribution_type: str = Field("normal", description="Distribution type")
    std_deviation: Optional[float] = Field(None, description="Standard deviation")


# -----------------------------------------------------------------------------
# Physics Explanations
# -----------------------------------------------------------------------------

class StoichiometryExplanation(BaseModel):
    """Explanation of combustion stoichiometry."""

    lambda_value: float = Field(..., ge=0.5, le=3.0, description="Air-fuel equivalence ratio")
    excess_air_percent: float = Field(..., description="Excess air percentage")
    oxygen_percent: float = Field(..., ge=0.0, le=21.0, description="Flue gas O2 percentage")
    stoichiometric_air: float = Field(..., description="Stoichiometric air (kg air/kg fuel)")
    actual_air: float = Field(..., description="Actual air supplied (kg air/kg fuel)")
    combustion_completeness: float = Field(..., ge=0.0, le=1.0, description="Combustion completeness")
    summary: str = Field(..., description="Plain language summary")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class EfficiencyExplanation(BaseModel):
    """Explanation of efficiency changes."""

    before_efficiency: float = Field(..., ge=0.0, le=1.0, description="Efficiency before change")
    after_efficiency: float = Field(..., ge=0.0, le=1.0, description="Efficiency after change")
    efficiency_delta: float = Field(..., description="Change in efficiency")
    primary_driver: str = Field(..., description="Primary driver of change")
    loss_breakdown: Dict[str, float] = Field(default_factory=dict, description="Loss breakdown")
    contributing_factors: List[FeatureContribution] = Field(default_factory=list)
    fuel_savings_percent: float = Field(..., description="Estimated fuel savings")
    summary: str = Field(..., description="Plain language summary")
    engineering_detail: str = Field(..., description="Technical explanation")


class EmissionExplanation(BaseModel):
    """Explanation of emission formation mechanisms."""

    emission_type: str = Field(..., description="Type of emission (CO, NOx, SOx, PM)")
    formation_mechanism: str = Field(..., description="Primary formation mechanism")
    current_level_ppm: float = Field(..., ge=0.0, description="Current emission level in ppm")
    regulatory_limit_ppm: float = Field(..., ge=0.0, description="Regulatory limit in ppm")
    margin_to_limit_percent: float = Field(..., description="Margin to regulatory limit")
    temperature_sensitivity: float = Field(..., description="Sensitivity to temperature (ppm/degC)")
    o2_sensitivity: float = Field(..., description="Sensitivity to O2 changes (ppm/%O2)")
    key_drivers: List[FeatureContribution] = Field(default_factory=list)
    reduction_strategies: List[str] = Field(default_factory=list)
    summary: str = Field(..., description="Plain language summary")
    engineering_detail: str = Field(..., description="Technical explanation")


class StabilityExplanation(BaseModel):
    """Explanation of combustion stability risks."""

    stability_index: float = Field(..., ge=0.0, le=1.0, description="Overall stability index")
    flame_stability_margin: float = Field(..., description="Margin to flame instability")
    pulsation_risk: float = Field(..., ge=0.0, le=1.0, description="Risk of combustion pulsation")
    flashback_risk: float = Field(..., ge=0.0, le=1.0, description="Risk of flame flashback")
    blowout_risk: float = Field(..., ge=0.0, le=1.0, description="Risk of flame blowout")
    risk_factors: List[FeatureContribution] = Field(default_factory=list)
    safe_operating_envelope: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    summary: str = Field(..., description="Plain language summary")
    engineering_detail: str = Field(..., description="Technical explanation")
    warnings: List[str] = Field(default_factory=list, description="Active warnings")


class PhysicsExplanation(BaseExplanation):
    """Complete physics-based explanation."""

    explanation_type: ExplanationType = Field(default=ExplanationType.PHYSICS, const=True)
    stoichiometry: Optional[StoichiometryExplanation] = None
    efficiency: Optional[EfficiencyExplanation] = None
    emissions: List[EmissionExplanation] = Field(default_factory=list)
    stability: Optional[StabilityExplanation] = None
    engineering_narrative: str = Field(..., description="Full engineering narrative")


# -----------------------------------------------------------------------------
# SHAP Explanations
# -----------------------------------------------------------------------------

class SHAPValues(BaseModel):
    """SHAP values for a model prediction."""

    base_value: float = Field(..., description="Expected value (mean prediction)")
    feature_names: List[str] = Field(..., description="Names of features")
    shap_values: List[float] = Field(..., description="SHAP values for each feature")
    feature_values: List[float] = Field(..., description="Actual values for each feature")
    prediction: float = Field(..., description="Model prediction")
    interaction_values: Optional[Dict[str, Dict[str, float]]] = None


class SHAPExplanation(BaseExplanation):
    """SHAP-based explanation for a prediction."""

    explanation_type: ExplanationType = Field(default=ExplanationType.SHAP, const=True)
    model_name: str = Field(..., description="Name of the model being explained")
    shap_values: SHAPValues = Field(..., description="SHAP values")
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    top_features: List[FeatureContribution] = Field(default_factory=list)
    interaction_effects: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# LIME Explanations
# -----------------------------------------------------------------------------

class LIMEExplanation(BaseExplanation):
    """LIME-based explanation for a prediction."""

    explanation_type: ExplanationType = Field(default=ExplanationType.LIME, const=True)
    model_name: str = Field(..., description="Name of the model being explained")
    local_prediction: float = Field(..., description="Prediction from local surrogate model")
    model_prediction: float = Field(..., description="Prediction from actual model")
    intercept: float = Field(..., description="Intercept of local linear model")
    feature_weights: Dict[str, float] = Field(default_factory=dict)
    feature_contributions: List[FeatureContribution] = Field(default_factory=list)
    local_r_squared: float = Field(..., ge=0.0, le=1.0, description="R-squared of local model")
    sample_size: int = Field(..., ge=1, description="Number of samples used for LIME")


class ConsistencyReport(BaseModel):
    """Report comparing SHAP and LIME explanations."""

    report_id: str = Field(..., description="Unique identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    shap_explanation_id: str = Field(..., description="ID of SHAP explanation")
    lime_explanation_id: str = Field(..., description="ID of LIME explanation")
    top_features_agreement: float = Field(..., ge=0.0, le=1.0)
    rank_correlation: float = Field(..., ge=-1.0, le=1.0)
    sign_agreement: float = Field(..., ge=0.0, le=1.0)
    magnitude_correlation: float = Field(..., ge=-1.0, le=1.0)
    is_consistent: bool = Field(..., description="Whether explanations are consistent")
    consistency_score: float = Field(..., ge=0.0, le=1.0)
    discrepancies: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class CounterfactualExplanation(BaseModel):
    """Counterfactual explanation showing how to achieve target."""

    explanation_id: str = Field(..., description="Unique identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    original_instance: Dict[str, float] = Field(..., description="Original feature values")
    original_prediction: float = Field(..., description="Prediction for original")
    target_prediction: float = Field(..., description="Target prediction to achieve")
    counterfactual_instance: Dict[str, float] = Field(..., description="Counterfactual values")
    counterfactual_prediction: float = Field(..., description="Prediction for counterfactual")
    changes_required: List[FeatureContribution] = Field(default_factory=list)
    feasibility_score: float = Field(..., ge=0.0, le=1.0)
    constraints_violated: List[str] = Field(default_factory=list)
    summary: str = Field(..., description="Plain language summary")


# -----------------------------------------------------------------------------
# Constraint Explanations
# -----------------------------------------------------------------------------

class ConstraintViolation(BaseModel):
    """Details of a constraint violation."""

    constraint_name: str = Field(..., description="Name of the constraint")
    constraint_type: str = Field(..., description="Type of constraint")
    current_value: float = Field(..., description="Current value")
    limit_value: float = Field(..., description="Constraint limit")
    violation_amount: float = Field(..., description="Amount of violation")
    violation_percent: float = Field(..., description="Violation as percentage")
    unit: str = Field(..., description="Engineering unit")


class ConstraintExplanation(BaseModel):
    """Explanation of a single constraint."""

    constraint_name: str = Field(..., description="Name of the constraint")
    constraint_type: str = Field(..., description="Type of constraint")
    current_value: float = Field(..., description="Current value")
    limit_value: float = Field(..., description="Constraint limit")
    status: ConstraintStatus = Field(..., description="Current constraint status")
    margin_to_limit: float = Field(..., description="Margin to limit (absolute)")
    margin_percent: float = Field(..., description="Margin to limit (percentage)")
    shadow_price: Optional[float] = None
    sensitivity: Optional[float] = None
    unit: str = Field(..., description="Engineering unit")
    physical_meaning: str = Field(..., description="Physical meaning")
    summary: str = Field(..., description="Plain language summary")
    engineering_detail: str = Field(..., description="Technical explanation")


class ViolationExplanation(BaseModel):
    """Detailed explanation of a constraint violation."""

    violation: ConstraintViolation = Field(..., description="Violation details")
    root_cause: str = Field(..., description="Root cause of violation")
    contributing_factors: List[FeatureContribution] = Field(default_factory=list)
    immediate_actions: List[str] = Field(default_factory=list)
    long_term_solutions: List[str] = Field(default_factory=list)
    safety_implications: Optional[str] = None
    summary: str = Field(..., description="Plain language summary")


class MarginExplanation(BaseModel):
    """Explanation of margin to a limit."""

    parameter_name: str = Field(..., description="Name of the parameter")
    current_value: float = Field(..., description="Current value")
    limit_value: float = Field(..., description="Limit value")
    margin_absolute: float = Field(..., description="Absolute margin")
    margin_percent: float = Field(..., description="Percentage margin")
    trend: ImpactDirection = Field(..., description="Current trend direction")
    time_to_limit: Optional[float] = None
    risk_level: str = Field(..., description="Risk level")
    recommendations: List[str] = Field(default_factory=list)
    summary: str = Field(..., description="Plain language summary")


class RelaxationSuggestion(BaseModel):
    """Suggestion for relaxing binding constraints."""

    constraint_name: str = Field(..., description="Name of the constraint")
    current_limit: float = Field(..., description="Current limit value")
    suggested_limit: float = Field(..., description="Suggested new limit")
    relaxation_amount: float = Field(..., description="Amount of relaxation")
    expected_benefit: float = Field(..., description="Expected improvement")
    benefit_per_unit_relaxation: float = Field(..., description="Shadow price")
    feasibility: str = Field(..., description="Feasibility assessment")
    risks: List[str] = Field(default_factory=list)
    approvals_required: List[str] = Field(default_factory=list)
    summary: str = Field(..., description="Plain language summary")


# -----------------------------------------------------------------------------
# Recommendation Explanations
# -----------------------------------------------------------------------------

class Recommendation(BaseModel):
    """A combustion optimization recommendation."""

    recommendation_id: str = Field(..., description="Unique identifier")
    parameter_name: str = Field(..., description="Parameter to adjust")
    current_value: float = Field(..., description="Current parameter value")
    recommended_value: float = Field(..., description="Recommended new value")
    change_amount: float = Field(..., description="Absolute change")
    change_percent: float = Field(..., description="Percentage change")
    unit: str = Field(..., description="Engineering unit")
    priority: int = Field(..., ge=1, le=5, description="Priority (1=highest)")
    rationale: str = Field(..., description="Reason for recommendation")
    expected_benefits: List[str] = Field(default_factory=list)
    implementation_time_minutes: float = Field(..., description="Time to implement")


class ImpactPrediction(BaseModel):
    """Predicted impact of implementing a recommendation."""

    recommendation_id: str = Field(..., description="ID of the recommendation")
    efficiency_change: float = Field(..., description="Predicted efficiency change")
    efficiency_change_bounds: UncertaintyBounds = Field(..., description="Uncertainty bounds")
    o2_change_percent: float = Field(..., description="Predicted O2 change")
    co_change_ppm: float = Field(..., description="Predicted CO change (ppm)")
    nox_change_ppm: float = Field(..., description="Predicted NOx change (ppm)")
    stability_margin_change: float = Field(..., description="Change in stability margin")
    fuel_savings_percent: float = Field(..., description="Predicted fuel savings")
    annual_savings_usd: Optional[float] = None
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in predictions")
    risk_factors: List[str] = Field(default_factory=list)


class RecommendationExplanation(BaseExplanation):
    """Complete explanation of a recommendation."""

    explanation_type: ExplanationType = Field(default=ExplanationType.RECOMMENDATION, const=True)
    recommendation: Recommendation = Field(..., description="The recommendation")
    impact_prediction: ImpactPrediction = Field(..., description="Predicted impact")
    physics_basis: str = Field(..., description="Physics-based explanation")
    model_basis: str = Field(..., description="ML model-based explanation")
    binding_constraints: List[str] = Field(default_factory=list)
    alternatives_considered: List[str] = Field(default_factory=list)
    operator_summary: str = Field(..., description="Plain language summary for operators")
    engineering_detail: str = Field(..., description="Technical detail for engineers")
    implementation_steps: List[str] = Field(default_factory=list)
    safety_checks: List[str] = Field(default_factory=list)


class ComparisonTable(BaseModel):
    """Comparison of multiple recommendation alternatives."""

    comparison_id: str = Field(..., description="Unique identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    recommendations: List[Recommendation] = Field(..., description="Recommendations compared")
    impact_predictions: List[ImpactPrediction] = Field(..., description="Impact predictions")
    ranking_criteria: List[str] = Field(..., description="Criteria used for ranking")
    ranked_order: List[str] = Field(..., description="Recommendation IDs in ranked order")
    tradeoff_analysis: str = Field(..., description="Analysis of tradeoffs")
    best_recommendation_id: str = Field(..., description="ID of best option")
    best_recommendation_rationale: str = Field(..., description="Rationale for best")


# -----------------------------------------------------------------------------
# Comprehensive Explanation
# -----------------------------------------------------------------------------

class ExplanationContext(BaseModel):
    """Context for generating a comprehensive explanation."""

    context_id: str = Field(..., description="Unique identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    boiler_id: str = Field(..., description="Boiler identifier")
    current_state: Dict[str, float] = Field(..., description="Current operating state")
    optimization_result: Dict[str, Any] = Field(..., description="Results from optimization")
    recommendations: List[Recommendation] = Field(default_factory=list)
    model_predictions: Dict[str, float] = Field(default_factory=dict)
    target_audience: AudienceLevel = Field(default=AudienceLevel.OPERATOR)
    include_physics: bool = Field(True)
    include_shap: bool = Field(True)
    include_lime: bool = Field(False)
    include_constraints: bool = Field(True)


class ComprehensiveExplanation(BaseExplanation):
    """Comprehensive explanation combining all explanation types."""

    explanation_type: ExplanationType = Field(default=ExplanationType.COMPREHENSIVE, const=True)
    context: ExplanationContext = Field(..., description="Context for this explanation")
    physics_explanation: Optional[PhysicsExplanation] = None
    shap_explanation: Optional[SHAPExplanation] = None
    lime_explanation: Optional[LIMEExplanation] = None
    constraint_explanations: List[ConstraintExplanation] = Field(default_factory=list)
    recommendation_explanations: List[RecommendationExplanation] = Field(default_factory=list)
    executive_summary: str = Field(..., description="High-level executive summary")
    operator_summary: str = Field(..., description="Plain language summary for operators")
    engineering_summary: str = Field(..., description="Technical summary for engineers")
    key_insights: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)


class OperatorExplanation(BaseModel):
    """Operator-friendly explanation format."""

    explanation_id: str = Field(..., description="ID of source explanation")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    title: str = Field(..., description="Explanation title")
    summary: str = Field(..., description="Plain language summary (2-3 sentences)")
    what_changed: List[str] = Field(default_factory=list)
    why_matters: str = Field(..., description="Why this matters")
    what_to_do: List[str] = Field(default_factory=list)
    expected_results: str = Field(..., description="Expected results")
    cautions: List[str] = Field(default_factory=list)
    confidence_statement: str = Field(..., description="Plain language confidence statement")


class EngineerExplanation(BaseModel):
    """Engineer-focused explanation format."""

    explanation_id: str = Field(..., description="ID of source explanation")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    title: str = Field(..., description="Explanation title")
    technical_summary: str = Field(..., description="Technical summary")
    physics_basis: str = Field(..., description="Physics-based reasoning")
    mathematical_model: str = Field(..., description="Mathematical formulation used")
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    constraint_analysis: List[ConstraintExplanation] = Field(default_factory=list)
    sensitivity_analysis: Dict[str, float] = Field(default_factory=dict)
    uncertainty_quantification: Dict[str, UncertaintyBounds] = Field(default_factory=dict)
    model_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    references: List[str] = Field(default_factory=list)
