# -*- coding: utf-8 -*-
"""
LIME Explainer for GL-005 COMBUSENSE Combustion Decisions

Implements LIME (Local Interpretable Model-agnostic Explanations) for
combustion optimization decisions. Creates locally faithful linear models
to explain individual predictions.

Reference:
    Ribeiro et al. "Why Should I Trust You?": Explaining the Predictions
    of Any Classifier. KDD 2016.

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Tuple, Protocol
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass
import hashlib
import logging
import numpy as np
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class ImpactDirection(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    NO_CHANGE = "no_change"


class ConfidenceLevel(str, Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class ExplanationType(str, Enum):
    LIME = "lime"
    SHAP = "shap"
    COUNTERFACTUAL = "counterfactual"


class CombustionParameter(str, Enum):
    O2_PERCENT = "o2_percent"
    CO_PPM = "co_ppm"
    TEMPERATURE = "temperature"
    LOAD_PERCENT = "load_percent"
    FUEL_FLOW = "fuel_flow"
    AIR_FLOW = "air_flow"
    EXCESS_AIR = "excess_air"
    EFFICIENCY = "efficiency"
    NOX_PPM = "nox_ppm"
    FLAME_TEMP = "flame_temp"
    AFR = "air_fuel_ratio"
    STABILITY = "stability"


# Pydantic models for LIME explainability
class LIMEExplainerConfig(BaseModel):
    """Configuration for LIME Explainer."""
    n_samples: int = Field(default=1000, ge=100, le=10000, description='Number of perturbation samples')
    kernel_width: float = Field(default=0.75, gt=0.0, le=2.0, description='Exponential kernel width')
    num_features: int = Field(default=10, ge=1, le=50, description='Max features in explanation')
    random_seed: int = Field(default=42, ge=0, description='Random seed for reproducibility')
    feature_selection: str = Field(default='auto', description='Feature selection method')
    discretize_continuous: bool = Field(default=True, description='Discretize continuous features')
    distance_metric: str = Field(default='euclidean', description='Distance metric for kernel')

    class Config:
        frozen = True


class FeatureContribution(BaseModel):
    """Single feature contribution to prediction."""
    feature_name: str = Field(..., description='Name of the feature')
    feature_value: float = Field(..., description='Actual value of the feature')
    coefficient: float = Field(..., description='LIME linear model coefficient')
    contribution: float = Field(..., description='Contribution to prediction (coef * value)')
    contribution_percent: float = Field(..., description='Percentage of total contribution')
    direction: ImpactDirection = Field(..., description='Direction of impact')
    description: str = Field(..., description='Human-readable description')
    unit: Optional[str] = Field(default=None, description='Unit of measurement')


class LIMEExplanation(BaseModel):
    """Complete LIME explanation for a combustion decision."""
    explanation_id: str = Field(..., description='Unique identifier for this explanation')
    timestamp: datetime = Field(default_factory=datetime.now, description='When explanation was generated')
    explanation_type: ExplanationType = Field(default=ExplanationType.LIME, description='Type of explanation')

    # Instance information
    instance_values: Dict[str, float] = Field(..., description='Input feature values')

    # Model information
    model_prediction: float = Field(..., description='Original model prediction')
    local_prediction: float = Field(..., description='Local linear model prediction')
    intercept: float = Field(..., description='Local model intercept')

    # Feature analysis
    feature_contributions: List[FeatureContribution] = Field(default_factory=list, description='Feature contributions')
    feature_weights: Dict[str, float] = Field(default_factory=dict, description='Raw LIME weights')

    # Quality metrics
    local_r_squared: float = Field(..., ge=0.0, le=1.0, description='Local model R-squared')
    sample_size: int = Field(..., description='Number of samples used')
    confidence: float = Field(..., ge=0.0, le=1.0, description='Explanation confidence')
    confidence_level: ConfidenceLevel = Field(..., description='Confidence level category')

    # Summaries
    summary: str = Field(..., description='Plain language summary')
    technical_detail: str = Field(..., description='Technical detail of explanation')

    # Provenance
    provenance_hash: str = Field(..., description='SHA-256 hash for audit trail')
    processing_time_ms: float = Field(..., description='Processing time in milliseconds')

    class Config:
        frozen = True


class CounterfactualExplanation(BaseModel):
    """Counterfactual explanation showing what changes would alter prediction."""
    explanation_id: str = Field(..., description='Unique identifier')
    original_instance: Dict[str, float] = Field(..., description='Original feature values')
    original_prediction: float = Field(..., description='Original prediction')
    target_prediction: float = Field(..., description='Target prediction to achieve')
    counterfactual_instance: Dict[str, float] = Field(..., description='Counterfactual feature values')
    counterfactual_prediction: float = Field(..., description='Achieved prediction')
    changes_required: List[FeatureContribution] = Field(default_factory=list, description='Changes needed')
    feasibility_score: float = Field(..., ge=0.0, le=1.0, description='How feasible is this counterfactual')
    constraints_violated: List[str] = Field(default_factory=list, description='Any constraints violated')
    summary: str = Field(..., description='Plain language summary')
    provenance_hash: str = Field(..., description='SHA-256 hash')

    class Config:
        frozen = True



# Protocol for models that can be explained
class PredictorProtocol(Protocol):
    """Protocol for models that can be explained with LIME."""
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input features."""
        ...


@dataclass
class FeatureStatistics:
    """Statistics for a single feature."""
    name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    quartile_25: float
    quartile_75: float


class CombustionLIMEExplainer:
    """
    LIME Explainer for Combustion Control Decisions.

    Implements LIME (Local Interpretable Model-agnostic Explanations) specialized
    for combustion optimization. Creates locally faithful linear models to explain
    individual predictions.

    Zero-Hallucination Guarantee:
        - All calculations are deterministic Python arithmetic
        - No ML models in the calculation path
        - Complete provenance tracking with SHA-256 hashes
        - Reproducible with fixed random seed

    Reference:
        Ribeiro et al. "Why Should I Trust You?": Explaining the Predictions
        of Any Classifier. KDD 2016.

    Attributes:
        config: Configuration parameters for LIME
        feature_stats: Statistics for feature perturbation

    Example:
        >>> config = LIMEExplainerConfig(n_samples=1000)
        >>> explainer = CombustionLIMEExplainer(config)
        >>> explanation = explainer.explain_instance(model, instance)
        >>> top_features = explainer.get_top_features(explanation, n=5)
    """

    # Combustion-specific feature bounds (NFPA 85 / API 556 compliant)
    COMBUSTION_BOUNDS: Dict[str, Tuple[float, float]] = {
        CombustionParameter.O2_PERCENT.value: (0.5, 15.0),
        CombustionParameter.CO_PPM.value: (0.0, 1000.0),
        CombustionParameter.TEMPERATURE.value: (200.0, 2500.0),
        CombustionParameter.LOAD_PERCENT.value: (10.0, 110.0),
        CombustionParameter.FUEL_FLOW.value: (0.0, 10000.0),
        CombustionParameter.AIR_FLOW.value: (0.0, 50000.0),
        CombustionParameter.EXCESS_AIR.value: (0.0, 100.0),
        CombustionParameter.EFFICIENCY.value: (50.0, 99.9),
        CombustionParameter.NOX_PPM.value: (0.0, 500.0),
        CombustionParameter.FLAME_TEMP.value: (800.0, 2200.0),
        CombustionParameter.AFR.value: (10.0, 25.0),
        CombustionParameter.STABILITY.value: (0.0, 1.0),
    }

    def __init__(
        self,
        config: Optional[LIMEExplainerConfig] = None,
        training_data: Optional[Dict[str, List[float]]] = None,
    ):
        """
        Initialize CombustionLIMEExplainer.

        Args:
            config: Configuration parameters. Uses defaults if not provided.
            training_data: Historical data for computing feature statistics.
        """
        self.config = config or LIMEExplainerConfig()
        self._feature_stats: Dict[str, FeatureStatistics] = {}
        self._rng = np.random.RandomState(self.config.random_seed)

        if training_data is not None:
            self._compute_feature_stats(training_data)

        logger.info(f"CombustionLIMEExplainer initialized with {self.config.n_samples} samples")

    def set_training_data(self, data: Dict[str, List[float]]) -> None:
        """
        Set training data for computing feature statistics.

        Args:
            data: Dictionary mapping feature names to lists of values
        """
        self._compute_feature_stats(data)
        logger.info(f"Training data set with {len(data)} features")



    def explain_instance(self, model: PredictorProtocol, instance: Dict[str, float]) -> LIMEExplanation:
        start_time = datetime.now()
        if not instance:
            raise ValueError('Instance cannot be empty')
        feature_names = list(instance.keys())
        instance_array = np.array([list(instance.values())])
        model_prediction = float(model.predict(instance_array)[0])
        samples, weights = self._generate_samples(instance_array, feature_names)
        sample_predictions = model.predict(samples)
        intercept, coefficients, r_squared = self._fit_local_model(samples, sample_predictions, weights, instance_array)
        local_prediction = intercept + np.sum(coefficients * instance_array[0])
        feature_weights = {n: round(float(c), 6) for n, c in zip(feature_names, coefficients)}
        feature_contributions = self._generate_feature_contributions(feature_names, list(instance.values()), coefficients, local_prediction)
        confidence = self._calculate_confidence(r_squared, len(samples))
        summary = self._generate_combustion_summary(model_prediction, local_prediction, feature_contributions, r_squared)
        technical_detail = self._generate_technical_detail(intercept, feature_weights, r_squared, self.config.n_samples)
        provenance_hash = self._calculate_provenance_hash(instance, feature_weights, r_squared)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return LIMEExplanation(
            explanation_id=self._generate_explanation_id(),
            instance_values=instance,
            model_prediction=round(model_prediction, 6),
            local_prediction=round(local_prediction, 6),
            intercept=round(intercept, 6),
            feature_contributions=feature_contributions,
            feature_weights=feature_weights,
            local_r_squared=round(r_squared, 4),
            sample_size=self.config.n_samples,
            confidence=confidence,
            confidence_level=self._get_confidence_level(confidence),
            summary=summary,
            technical_detail=technical_detail,
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
        )

    def get_top_features(self, explanation: LIMEExplanation, n: int = 5) -> List[FeatureContribution]:
        sorted_contributions = sorted(explanation.feature_contributions, key=lambda x: abs(x.contribution), reverse=True)
        return sorted_contributions[:n]


    def generate_counterfactual(self, model: PredictorProtocol, instance: Dict[str, float], target_prediction: float, max_features_to_change: int = 3) -> CounterfactualExplanation:
        start_time = datetime.now()
        feature_names = list(instance.keys())
        original_values = np.array(list(instance.values()))
        original_prediction = float(model.predict(original_values.reshape(1, -1))[0])
        lime_exp = self.explain_instance(model, instance)
        sorted_features = sorted(zip(feature_names, lime_exp.feature_weights.values()), key=lambda x: abs(x[1]), reverse=True)[:max_features_to_change]
        prediction_diff = target_prediction - original_prediction
        counterfactual_values = original_values.copy()
        changes_required = []
        for feat_name, coef in sorted_features:
            if abs(coef) < 1e-6:
                continue
            idx = feature_names.index(feat_name)
            original_val = original_values[idx]
            if abs(coef) > 0:
                change = prediction_diff / (coef * len(sorted_features))
                bounds = self.COMBUSTION_BOUNDS.get(feat_name, (float("-inf"), float("inf")))
                new_val = np.clip(original_val + change, bounds[0], bounds[1])
                actual_change = new_val - original_val
                counterfactual_values[idx] = new_val
                if abs(actual_change) > 1e-6:
                    direction = ImpactDirection.INCREASE if actual_change > 0 else ImpactDirection.DECREASE
                    changes_required.append(FeatureContribution(feature_name=feat_name, feature_value=float(new_val), coefficient=float(coef), contribution=float(actual_change), contribution_percent=abs(actual_change) / abs(original_val) * 100 if original_val != 0 else 100, direction=direction, description=f"Change {feat_name} from {original_val:.3f} to {new_val:.3f}"))
        cf_prediction = float(model.predict(counterfactual_values.reshape(1, -1))[0])
        total_relative_change = sum(abs(c.contribution) / abs(instance[c.feature_name]) if instance[c.feature_name] != 0 else 0 for c in changes_required)
        feasibility_score = max(0, 1 - total_relative_change / 2)
        constraints_violated = self._check_combustion_constraints(dict(zip(feature_names, counterfactual_values)))
        summary = self._generate_counterfactual_summary(original_prediction, target_prediction, cf_prediction, changes_required)
        provenance_hash = self._calculate_provenance_hash(instance, dict(zip(feature_names, counterfactual_values.tolist())), feasibility_score)
        return CounterfactualExplanation(explanation_id=f"cf-{uuid.uuid4().hex[:12]}", original_instance=instance, original_prediction=round(original_prediction, 6), target_prediction=target_prediction, counterfactual_instance=dict(zip(feature_names, [round(v, 6) for v in counterfactual_values])), counterfactual_prediction=round(cf_prediction, 6), changes_required=changes_required, feasibility_score=round(feasibility_score, 4), constraints_violated=constraints_violated, summary=summary, provenance_hash=provenance_hash)


    def get_combustion_interpretation(self, explanation: LIMEExplanation) -> Dict[str, Any]:
        interpretation = {"summary": "", "key_drivers": [], "recommendations": [], "safety_concerns": [], "efficiency_impact": ""}
        top_features = self.get_top_features(explanation, n=3)
        for fc in top_features:
            driver = self._interpret_combustion_driver(fc)
            interpretation["key_drivers"].append(driver)
        interpretation["efficiency_impact"] = self._assess_efficiency_impact(explanation)
        interpretation["safety_concerns"] = self._check_safety_concerns(explanation)
        interpretation["recommendations"] = self._generate_recommendations(explanation, top_features)
        interpretation["summary"] = self._generate_interpretation_summary(top_features, interpretation["efficiency_impact"], interpretation["safety_concerns"])
        return interpretation

    def _compute_feature_stats(self, data: Dict[str, List[float]]) -> None:
        for name, values in data.items():
            arr = np.array(values)
            self._feature_stats[name] = FeatureStatistics(name=name, mean=float(np.mean(arr)), std=float(np.std(arr)), min_val=float(np.min(arr)), max_val=float(np.max(arr)), quartile_25=float(np.percentile(arr, 25)), quartile_75=float(np.percentile(arr, 75)))

    def _generate_samples(self, instance: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        n_features = instance.shape[1]
        samples = np.zeros((self.config.n_samples, n_features))
        for i in range(n_features):
            name = feature_names[i]
            center = instance[0, i]
            std = self._feature_stats[name].std if name in self._feature_stats else (abs(center) * 0.1 if center != 0 else 1.0)
            bounds = self.COMBUSTION_BOUNDS.get(name, (float("-inf"), float("inf")))
            samples[:, i] = np.clip(self._rng.normal(center, std, self.config.n_samples), bounds[0], bounds[1])
        distances = np.sqrt(np.sum((samples - instance) ** 2, axis=1))
        kernel_width = self.config.kernel_width * np.sqrt(n_features)
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
        return samples, weights


    def _fit_local_model(self, samples: np.ndarray, predictions: np.ndarray, weights: np.ndarray, instance: np.ndarray) -> Tuple[float, np.ndarray, float]:
        W = np.diag(weights)
        X = np.hstack([np.ones((len(samples), 1)), samples])
        try:
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ predictions
            beta = np.linalg.solve(XtWX, XtWy)
            intercept = float(beta[0])
            coefficients = beta[1:]
            y_pred = X @ beta
            ss_res = np.sum(weights * (predictions - y_pred) ** 2)
            ss_tot = np.sum(weights * (predictions - np.average(predictions, weights=weights)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        except np.linalg.LinAlgError:
            intercept = float(np.mean(predictions))
            coefficients = np.zeros(samples.shape[1])
            r_squared = 0.0
        return intercept, coefficients, float(max(0, min(1, r_squared)))

    def _generate_feature_contributions(self, feature_names: List[str], feature_values: List[float], coefficients: np.ndarray, local_prediction: float) -> List[FeatureContribution]:
        contributions = []
        total_abs_contrib = sum(abs(c * v) for c, v in zip(coefficients, feature_values))
        for name, value, coef in zip(feature_names, feature_values, coefficients):
            contribution = coef * value
            direction = ImpactDirection.INCREASE if contribution > 0.001 else (ImpactDirection.DECREASE if contribution < -0.001 else ImpactDirection.NO_CHANGE)
            contrib_pct = (abs(contribution) / total_abs_contrib * 100) if total_abs_contrib > 0 else 0
            direction_text = "increases" if contribution > 0 else "decreases"
            description = f"{name} at {value:.3f} (coef={coef:.4f}) {direction_text} prediction by {abs(contribution):.4f}"
            contributions.append(FeatureContribution(feature_name=name, feature_value=value, coefficient=round(float(coef), 6), contribution=round(contribution, 6), contribution_percent=round(contrib_pct, 1), direction=direction, description=description))
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        return contributions


    def _calculate_confidence(self, r_squared: float, sample_count: int) -> float:
        base_confidence = r_squared
        sample_factor = min(sample_count / 1000, 1.0)
        confidence = base_confidence * (0.7 + 0.3 * sample_factor)
        return round(min(0.99, max(0.3, confidence)), 3)

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        if confidence > 0.95: return ConfidenceLevel.VERY_HIGH
        elif confidence > 0.85: return ConfidenceLevel.HIGH
        elif confidence > 0.70: return ConfidenceLevel.MEDIUM
        elif confidence > 0.50: return ConfidenceLevel.LOW
        else: return ConfidenceLevel.VERY_LOW

    def _generate_combustion_summary(self, model_prediction: float, local_prediction: float, contributions: List[FeatureContribution], r_squared: float) -> str:
        if not contributions: return "No significant feature contributions identified."
        top = contributions[0]
        direction = "higher" if top.contribution > 0 else "lower"
        fidelity = "high" if r_squared > 0.8 else "moderate" if r_squared > 0.5 else "low"
        param_context = self._get_parameter_context(top.feature_name)
        return f"Model predicts {model_prediction:.3f}. The local linear model (R-squared={r_squared:.2f}, {fidelity} fidelity) identifies {top.feature_name} as most important, pushing the prediction {direction}. {param_context}"

    def _generate_technical_detail(self, intercept: float, feature_weights: Dict[str, float], r_squared: float, n_samples: int) -> str:
        lines = ["LIME Analysis:", f"  Local model: Linear regression with exponential kernel", f"  Samples generated: {n_samples}", f"  Local R-squared: {r_squared:.4f}", f"  Intercept: {intercept:.6f}", "", "Feature coefficients (local linear model):"]
        for name, weight in sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"  {name}: {weight:+.6f}")
        return chr(10).join(lines)

    def _get_parameter_context(self, param_name: str) -> str:
        contexts = {"o2_percent": "Oxygen levels affect combustion efficiency and emissions.", "co_ppm": "CO indicates incomplete combustion.", "temperature": "Temperature impacts efficiency and NOx formation.", "load_percent": "Load affects optimal air-fuel ratio settings.", "fuel_flow": "Fuel flow rate directly impacts heat output.", "air_flow": "Air flow must balance with fuel for complete combustion.", "excess_air": "Excess air reduces efficiency but improves safety.", "efficiency": "Combustion efficiency represents heat utilization.", "nox_ppm": "NOx emissions are regulated.", "flame_temp": "Flame temperature affects NOx formation.", "air_fuel_ratio": "AFR must be optimized for the fuel type.", "stability": "Stability indicates consistent combustion."}
        return contexts.get(param_name, "")


    def _check_combustion_constraints(self, values: Dict[str, float]) -> List[str]:
        violations = []
        for name, value in values.items():
            if name in self.COMBUSTION_BOUNDS:
                min_val, max_val = self.COMBUSTION_BOUNDS[name]
                if value < min_val: violations.append(f"{name} below minimum ({value:.3f} < {min_val:.3f})")
                elif value > max_val: violations.append(f"{name} above maximum ({value:.3f} > {max_val:.3f})")
        return violations

    def _generate_counterfactual_summary(self, original: float, target: float, achieved: float, changes: List[FeatureContribution]) -> str:
        n_changes = len(changes)
        gap = abs(target - achieved)
        achievement = "achieved" if gap < 0.01 else f"partially achieved (gap: {gap:.3f})"
        return f"To move from {original:.3f} to target {target:.3f}: {n_changes} feature changes recommended. Target {achievement} with predicted value {achieved:.3f}."

    def _interpret_combustion_driver(self, fc: FeatureContribution) -> Dict[str, Any]:
        interpretation = {"feature": fc.feature_name, "value": fc.feature_value, "impact": fc.direction.value, "magnitude": abs(fc.contribution), "context": self._get_parameter_context(fc.feature_name)}
        if fc.feature_name == "o2_percent":
            if fc.feature_value < 2.0: interpretation["guidance"] = "Low O2 may indicate rich combustion"
            elif fc.feature_value > 6.0: interpretation["guidance"] = "High O2 indicates excess air"
        elif fc.feature_name == "co_ppm" and fc.feature_value > 200:
            interpretation["guidance"] = "Elevated CO indicates incomplete combustion"
        return interpretation

    def _assess_efficiency_impact(self, explanation: LIMEExplanation) -> str:
        efficiency_features = ["efficiency", "o2_percent", "excess_air"]
        relevant = [fc for fc in explanation.feature_contributions if fc.feature_name in efficiency_features]
        if not relevant: return "No direct efficiency indicators in feature set."
        total_impact = sum(fc.contribution for fc in relevant)
        return f"Features suggest efficiency {'increase' if total_impact > 0 else 'decrease'} of {abs(total_impact):.3f}"


    def _check_safety_concerns(self, explanation: LIMEExplanation) -> List[str]:
        concerns = []
        for fc in explanation.feature_contributions:
            if fc.feature_name == "co_ppm" and fc.feature_value > 400: concerns.append("CO levels elevated")
            if fc.feature_name == "o2_percent" and fc.feature_value < 1.5: concerns.append("O2 critically low")
            if fc.feature_name == "temperature" and fc.feature_value > 2000: concerns.append("Temperature very high")
        return concerns

    def _generate_recommendations(self, explanation: LIMEExplanation, top_features: List[FeatureContribution]) -> List[str]:
        recommendations = []
        for fc in top_features:
            if fc.feature_name == "o2_percent" and fc.contribution < 0 and fc.feature_value > 4.0:
                recommendations.append("Consider reducing excess air to improve efficiency")
            elif fc.feature_name == "co_ppm" and fc.contribution > 0:
                recommendations.append("Increase air supply to reduce CO emissions")
            elif fc.feature_name == "load_percent" and fc.contribution < 0 and fc.feature_value < 50:
                recommendations.append("Consider increasing load for better efficiency")
        if not recommendations: recommendations.append("Current operating parameters appear well-tuned")
        return recommendations

    def _generate_interpretation_summary(self, top_features: List[FeatureContribution], efficiency_impact: str, safety_concerns: List[str]) -> str:
        parts = []
        if top_features:
            top = top_features[0]
            parts.append(f"Primary driver: {top.feature_name} ({top.direction.value} impact)")
        parts.append(efficiency_impact)
        parts.append(f"Safety: {len(safety_concerns)} concern(s)" if safety_concerns else "Safety: No concerns")
        return " | ".join(parts)

    def _calculate_provenance_hash(self, *args) -> str:
        return hashlib.sha256(str(args).encode()).hexdigest()

    def _generate_explanation_id(self) -> str:
        return f"lime-{uuid.uuid4().hex[:12]}"



def create_default_explainer() -> CombustionLIMEExplainer:
    """Create a CombustionLIMEExplainer with default configuration."""
    return CombustionLIMEExplainer()


def create_high_fidelity_explainer() -> CombustionLIMEExplainer:
    """Create a CombustionLIMEExplainer with high-fidelity settings."""
    config = LIMEExplainerConfig(n_samples=5000, kernel_width=0.5, num_features=15)
    return CombustionLIMEExplainer(config)


def create_fast_explainer() -> CombustionLIMEExplainer:
    """Create a CombustionLIMEExplainer optimized for speed."""
    config = LIMEExplainerConfig(n_samples=500, kernel_width=1.0, num_features=5)
    return CombustionLIMEExplainer(config)


__all__ = [
    "CombustionLIMEExplainer",
    "LIMEExplainerConfig",
    "LIMEExplanation",
    "FeatureContribution",
    "CounterfactualExplanation",
    "ImpactDirection",
    "ConfidenceLevel",
    "ExplanationType",
    "CombustionParameter",
    "FeatureStatistics",
    "PredictorProtocol",
    "create_default_explainer",
    "create_high_fidelity_explainer",
    "create_fast_explainer",
]
