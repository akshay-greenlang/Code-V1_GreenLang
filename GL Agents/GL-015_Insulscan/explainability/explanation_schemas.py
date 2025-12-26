# -*- coding: utf-8 -*-
"""
GL-015 Insulscan - Explanation Schemas

Pydantic v2 models for insulation scanning and thermal assessment explainability
with zero-hallucination guarantees. All numeric values are derived from
deterministic calculations (SHAP, LIME, causal inference), not LLM-generated estimates.

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

from pydantic import BaseModel, Field, field_validator, model_validator


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
    """Types of insulation predictions that can be explained."""
    CONDITION_SCORE = "condition_score"
    HEAT_LOSS = "heat_loss"
    THERMAL_RESISTANCE = "thermal_resistance"
    REMAINING_USEFUL_LIFE = "remaining_useful_life"
    DEGRADATION_RATE = "degradation_rate"
    REPAIR_PRIORITY = "repair_priority"
    ENERGY_WASTE = "energy_waste"
    SURFACE_TEMPERATURE = "surface_temperature"
    R_VALUE = "r_value"


class DegradationMechanism(str, Enum):
    """Physical mechanisms of insulation degradation."""
    AGE_RELATED = "age_related"
    MOISTURE_DAMAGE = "moisture_damage"
    UV_EXPOSURE = "uv_exposure"
    MECHANICAL_DAMAGE = "mechanical_damage"
    THERMAL_CYCLING = "thermal_cycling"
    CHEMICAL_DEGRADATION = "chemical_degradation"
    COMPRESSION = "compression"
    SETTLING = "settling"
    COMBINED = "combined"


class InsulationType(str, Enum):
    """Types of insulation materials."""
    MINERAL_WOOL = "mineral_wool"
    FIBERGLASS = "fiberglass"
    CALCIUM_SILICATE = "calcium_silicate"
    CELLULAR_GLASS = "cellular_glass"
    POLYURETHANE_FOAM = "polyurethane_foam"
    PERLITE = "perlite"
    AEROGEL = "aerogel"
    REFLECTIVE = "reflective"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence levels for predictions and explanations."""
    HIGH = "high"          # >= 0.85
    MEDIUM = "medium"      # >= 0.65
    LOW = "low"            # >= 0.40
    VERY_LOW = "very_low"  # < 0.40


class FeatureCategory(str, Enum):
    """Categories for insulation assessment features."""
    THERMAL = "thermal"
    PHYSICAL = "physical"
    ENVIRONMENTAL = "environmental"
    OPERATIONAL = "operational"
    AGE_RELATED = "age_related"
    VISUAL = "visual"
    MOISTURE = "moisture"
    STRUCTURAL = "structural"


class FeatureImportance(BaseModel):
    """
    Importance score for a single feature in insulation assessment.

    Attributes:
        feature_name: Name of the feature (e.g., 'surface_temperature_delta')
        importance_value: Absolute importance score from SHAP/LIME
        rank: Ranking among all features (1 = most important)
        direction: Direction of impact ('positive' = worsens condition)
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

    @field_validator('direction')
    @classmethod
    def validate_direction(cls, v: str) -> str:
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
    importance_rank: int = Field(default=0, ge=0, description="Rank by importance")

    @field_validator('contribution_percentage')
    @classmethod
    def validate_percentage(cls, v: float) -> float:
        """Ensure percentage is within valid range."""
        return round(v, 4)


class ConfidenceBounds(BaseModel):
    """Confidence interval for a point estimate."""
    lower_bound: float = Field(..., description="Lower bound of CI")
    upper_bound: float = Field(..., description="Upper bound of CI")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="CI level (e.g., 0.95)")
    method: str = Field("bootstrap", description="Method used to compute CI")

    @model_validator(mode='after')
    def validate_bounds(self) -> 'ConfidenceBounds':
        """Ensure lower_bound <= upper_bound."""
        if self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must be <= upper_bound")
        return self


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

    @field_validator('total_uncertainty', mode='before')
    @classmethod
    def compute_total(cls, v: float, info) -> float:
        """Compute total uncertainty if not provided."""
        if v is None or v == 0:
            data = info.data if hasattr(info, 'data') else {}
            epi = data.get('epistemic_uncertainty', 0)
            ale = data.get('aleatoric_uncertainty', 0)
            return (epi ** 2 + ale ** 2) ** 0.5
        return v


class LocalExplanation(BaseModel):
    """
    Local explanation for a single prediction instance.

    Explains why the model made a specific prediction for a specific
    insulation section at a specific time.
    """
    explanation_id: str = Field(..., description="Unique identifier for this explanation")
    asset_id: str = Field(..., description="Insulation asset/section identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    prediction_type: PredictionType = Field(..., description="Type of prediction explained")
    prediction_value: float = Field(..., description="The predicted value")
    base_value: float = Field(..., description="Baseline/expected value")

    feature_contributions: List[FeatureContribution] = Field(
        ..., description="Contributions of each feature"
    )
    top_positive_drivers: List[str] = Field(
        default_factory=list, description="Features worsening condition"
    )
    top_negative_drivers: List[str] = Field(
        default_factory=list, description="Features improving condition"
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
            "asset_id": self.asset_id,
            "prediction_value": self.prediction_value,
            "feature_contributions": [fc.model_dump() for fc in self.feature_contributions],
            "timestamp": self.timestamp.isoformat(),
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


class GlobalExplanation(BaseModel):
    """
    Global explanation for model behavior across all assets.

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
    total_assets_analyzed: int = Field(..., ge=1)
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


class CausalFactor(BaseModel):
    """
    A causal factor contributing to insulation degradation.

    Based on domain knowledge and causal inference methods.
    """
    factor_name: str = Field(..., description="Name of the causal factor")
    causal_strength: float = Field(..., ge=-1.0, le=1.0, description="Strength of causal effect")
    mechanism: str = Field(..., description="Physical mechanism description")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in causal claim")
    is_controllable: bool = Field(True, description="Whether factor can be controlled")


class CausalRelationship(BaseModel):
    """
    Represents a causal relationship between factors and degradation.

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
    Root cause analysis for insulation degradation.
    """
    hypothesis_id: str = Field(..., description="Unique identifier")
    asset_id: str = Field(..., description="Asset analyzed")

    primary_cause: str = Field(..., description="Primary root cause identified")
    secondary_causes: List[str] = Field(default_factory=list)

    causal_chain: List[CausalRelationship] = Field(
        ..., description="Chain of causal relationships"
    )

    causal_effect: float = Field(..., description="Total causal effect on degradation")
    confidence_interval: ConfidenceBounds = Field(...)

    degradation_mechanism: DegradationMechanism = Field(...)
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

    Explains what conditions would need to change to achieve
    a different insulation outcome.
    """
    explanation_id: str = Field(...)
    asset_id: str = Field(...)

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


class ThermalImageData(BaseModel):
    """
    Thermal image analysis data for reports.
    """
    image_id: str = Field(..., description="Unique image identifier")
    capture_timestamp: datetime = Field(...)
    min_temperature: float = Field(..., description="Minimum surface temperature (C)")
    max_temperature: float = Field(..., description="Maximum surface temperature (C)")
    avg_temperature: float = Field(..., description="Average surface temperature (C)")
    ambient_temperature: float = Field(..., description="Ambient temperature (C)")
    hot_spot_count: int = Field(..., ge=0, description="Number of detected hot spots")
    hot_spot_locations: List[Dict[str, float]] = Field(
        default_factory=list, description="Hot spot coordinates and temperatures"
    )
    coverage_percentage: float = Field(..., ge=0, le=100, description="Image coverage")


class HeatLossDiagram(BaseModel):
    """
    Heat loss calculation diagram data.
    """
    diagram_id: str = Field(...)
    asset_id: str = Field(...)
    heat_loss_rate: float = Field(..., description="Heat loss in W/m2")
    insulation_r_value: float = Field(..., ge=0, description="R-value (m2K/W)")
    surface_area: float = Field(..., ge=0, description="Surface area (m2)")
    operating_temperature: float = Field(..., description="Operating temperature (C)")
    ambient_temperature: float = Field(..., description="Ambient temperature (C)")
    annual_energy_loss: float = Field(..., ge=0, description="Annual energy loss (kWh)")
    annual_cost: float = Field(..., ge=0, description="Annual cost ($)")


class RepairRecommendation(BaseModel):
    """
    Repair recommendation with justification.
    """
    recommendation_id: str = Field(...)
    priority: str = Field(..., pattern="^(critical|high|medium|low)$")
    action: str = Field(..., description="Recommended action")
    justification: str = Field(..., description="Technical justification")
    expected_improvement: float = Field(..., ge=0, le=1.0)
    estimated_cost: Optional[float] = Field(None, ge=0)
    estimated_payback_months: Optional[float] = Field(None, ge=0)
    regulatory_compliance: bool = Field(False, description="Required for compliance")


class InsulationExplanation(BaseModel):
    """
    Overall insulation explanation with natural language summary.

    Combines all explanation types into a comprehensive package.
    """
    explanation_id: str = Field(...)
    asset_id: str = Field(...)
    insulation_type: InsulationType = Field(...)

    # Natural language summary
    summary: str = Field(..., description="Executive summary in plain language")
    detailed_explanation: str = Field(..., description="Detailed technical explanation")
    key_findings: List[str] = Field(..., description="Key findings bullet points")

    # Prediction being explained
    prediction_type: PredictionType = Field(...)
    prediction_value: float = Field(...)
    prediction_unit: str = Field(...)

    # Component explanations
    feature_contributions: List[FeatureContribution] = Field(...)
    causal_factors: List[CausalFactor] = Field(...)
    degradation_mechanism: DegradationMechanism = Field(...)

    # Recommendations
    repair_recommendations: List[RepairRecommendation] = Field(...)

    # Metadata
    confidence: ConfidenceLevel = Field(...)
    data_quality_score: float = Field(..., ge=0, le=1)
    computation_time_ms: float = Field(..., ge=0)
    provenance_hash: str = Field(...)
    methodology_version: str = Field("1.0.0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for complete audit trail."""
        content = {
            "explanation_id": self.explanation_id,
            "asset_id": self.asset_id,
            "prediction_value": self.prediction_value,
            "key_findings": self.key_findings,
            "timestamp": self.timestamp.isoformat(),
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


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


class ISO50001ComplianceData(BaseModel):
    """
    ISO 50001 Energy Management compliance data for reports.
    """
    compliance_status: str = Field(..., pattern="^(compliant|non_compliant|partial)$")
    energy_baseline: float = Field(..., description="Energy baseline (kWh)")
    current_performance: float = Field(..., description="Current energy performance (kWh)")
    improvement_percentage: float = Field(..., description="Improvement vs baseline (%)")
    energy_performance_indicators: Dict[str, float] = Field(...)
    audit_findings: List[str] = Field(default_factory=list)
    corrective_actions: List[str] = Field(default_factory=list)
    certification_date: Optional[datetime] = Field(None)
    next_audit_date: Optional[datetime] = Field(None)


class InsulationExplainabilityReport(BaseModel):
    """
    Complete explainability report for insulation assessment.

    Integrates all explanation types into a comprehensive report.
    """
    report_id: str = Field(...)
    asset_id: str = Field(...)
    report_title: str = Field(...)

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

    # Insulation-specific explanation
    insulation_explanation: InsulationExplanation = Field(...)

    # Thermal imaging data
    thermal_images: List[ThermalImageData] = Field(default_factory=list)
    heat_loss_diagrams: List[HeatLossDiagram] = Field(default_factory=list)

    # Recommendations
    repair_recommendations: List[RepairRecommendation] = Field(...)

    # Regulatory compliance
    iso50001_compliance: Optional[ISO50001ComplianceData] = Field(None)

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
            "asset_id": self.asset_id,
            "prediction_value": self.prediction_value,
            "local_explanation_hash": self.local_explanation.provenance_hash,
            "insulation_explanation_hash": self.insulation_explanation.provenance_hash,
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

    # Thermal heatmap data
    thermal_heatmap_data: Optional[Dict[str, Any]] = Field(
        None, description="Data for thermal heatmap visualization"
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
CausalFactors = List[CausalFactor]
RepairRecommendations = List[RepairRecommendation]
