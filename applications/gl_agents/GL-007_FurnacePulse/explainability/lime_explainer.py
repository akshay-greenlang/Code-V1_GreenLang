"""
GL-007 FurnacePulse - LIME Explainer

Local Interpretable Model-agnostic Explanations for furnace
monitoring predictions. Provides local surrogate explanations
around specific sensor reading points.

This module provides:
- Local interpretable explanations for individual predictions
- Tabular data explanations for sensor readings
- Confidence scoring for explanation quality
- Complementary validation to SHAP explanations

Reference: Ribeiro et al., "Why Should I Trust You?:
Explaining the Predictions of Any Classifier", KDD 2016.

Zero-Hallucination: LIME explanations are derived from
local linear approximations, not generative AI outputs.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
import hashlib
import json
import logging
import math

import numpy as np

try:
    from lime import lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Type of LIME explanation."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ConfidenceLevel(Enum):
    """Confidence level of explanation."""
    HIGH = "high"      # R2 >= 0.8
    MEDIUM = "medium"  # 0.5 <= R2 < 0.8
    LOW = "low"        # R2 < 0.5


@dataclass
class TabularExplanation:
    """Explanation for tabular sensor data."""

    feature_name: str
    feature_value: float
    weight: float
    direction: str  # "positive" or "negative"
    contribution_text: str
    discretized_range: Optional[str] = None


@dataclass
class LIMEResult:
    """Result from LIME analysis for a single prediction."""

    prediction_type: str
    prediction_id: str
    feature_names: List[str]
    feature_weights: Dict[str, float]
    feature_values: Dict[str, float]
    local_prediction: float
    intercept: float
    score: float  # R2 of local surrogate
    confidence_level: ConfidenceLevel
    num_features_used: int
    explanation_text: List[str]
    tabular_explanations: List[TabularExplanation]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    computation_hash: str = ""
    model_version: str = ""


@dataclass
class ComparisonResult:
    """Result comparing LIME and SHAP explanations."""

    agreement_score: float  # 0-1 how much LIME/SHAP agree
    top_features_overlap: List[str]
    disagreements: List[Dict[str, Any]]
    recommendation: str
    combined_confidence: float


class LIMEExplainer:
    """
    LIME-based explainer for furnace monitoring predictions.

    Creates local linear approximations around specific sensor
    reading points to explain why predictions were made. Complements
    SHAP analysis with an alternative explanation approach.

    Use cases:
    - Explain why a specific hotspot was detected
    - Understand local sensitivity to sensor changes
    - Validate SHAP explanations with alternative method
    - Provide human-readable feature contribution text

    Example:
        >>> explainer = LIMEExplainer(model_version="v2.1.0")
        >>> result = explainer.explain_prediction(
        ...     sensor_readings, prediction_value, prediction_fn
        ... )
        >>> for line in result.explanation_text:
        ...     print(line)
        >>> print(f"Explanation confidence: {result.confidence_level.value}")
    """

    VERSION = "1.0.0"

    # Feature display names for human-readable explanations
    FEATURE_DISPLAY_NAMES = {
        "tmt": "Tube Metal Temperature",
        "flame_intensity": "Flame Intensity",
        "flame_stability": "Flame Stability Index",
        "flow_rate": "Process Flow Rate",
        "flue_temp": "Flue Gas Temperature",
        "pressure": "Pressure",
        "o2": "Oxygen Level",
        "co2": "CO2 Level",
        "nox": "NOx Emissions",
        "vibration": "Vibration Amplitude",
    }

    # Units for features
    FEATURE_UNITS = {
        "tmt": "C",
        "flame_intensity": "%",
        "flame_stability": "",
        "flow_rate": "kg/s",
        "flue_temp": "C",
        "pressure": "bar",
        "o2": "%",
        "co2": "%",
        "nox": "ppm",
        "vibration": "mm/s",
    }

    def __init__(
        self,
        model_version: str = "1.0.0",
        num_features: int = 10,
        num_samples: int = 500,
        kernel_width: float = 0.75,
        discretize_continuous: bool = True,
        random_state: int = 42,
    ) -> None:
        """
        Initialize LIME explainer for furnace predictions.

        Args:
            model_version: Version of the prediction model
            num_features: Number of features in explanation
            num_samples: Number of perturbed samples for LIME
            kernel_width: Width of exponential kernel
            discretize_continuous: Whether to discretize continuous features
            random_state: Random seed for reproducibility
        """
        self.model_version = model_version
        self.num_features = num_features
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.discretize_continuous = discretize_continuous
        self.random_state = random_state
        self._explainer = None
        self._feature_names: List[str] = []
        self._training_data: Optional[np.ndarray] = None

    def explain_prediction(
        self,
        sensor_readings: Dict[str, float],
        predicted_value: float,
        prediction_fn: Callable[[np.ndarray], np.ndarray],
        prediction_type: str = "hotspot",
        prediction_id: Optional[str] = None,
    ) -> LIMEResult:
        """
        Explain a single prediction using LIME.

        Args:
            sensor_readings: Current sensor values
            predicted_value: The prediction to explain
            prediction_fn: Function that takes features and returns predictions
            prediction_type: Type of prediction (hotspot, efficiency, rul)
            prediction_id: Optional ID for the prediction

        Returns:
            LIMEResult with local explanation
        """
        if prediction_id is None:
            prediction_id = self._generate_id()

        # Extract features
        features, feature_names = self._extract_features(sensor_readings)
        self._feature_names = feature_names

        # Generate training data for LIME
        if self._training_data is None:
            self._training_data = self._generate_training_data(features)

        # Calculate LIME explanation
        if HAS_LIME:
            result = self._calculate_lime_explanation(
                features, feature_names, prediction_fn
            )
        else:
            result = self._calculate_fallback_explanation(
                features, feature_names, prediction_fn
            )

        # Determine confidence level
        confidence_level = self._determine_confidence(result["score"])

        # Generate tabular explanations
        tabular_explanations = self._generate_tabular_explanations(
            result["weights"], sensor_readings
        )

        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            result["weights"], sensor_readings, prediction_type
        )

        # Compute hash
        computation_hash = self._compute_hash(
            features, result["weights"], predicted_value
        )

        return LIMEResult(
            prediction_type=prediction_type,
            prediction_id=prediction_id,
            feature_names=feature_names,
            feature_weights=result["weights"],
            feature_values=sensor_readings,
            local_prediction=round(result["local_prediction"], 4),
            intercept=round(result["intercept"], 4),
            score=round(result["score"], 4),
            confidence_level=confidence_level,
            num_features_used=len(result["weights"]),
            explanation_text=explanation_text,
            tabular_explanations=tabular_explanations,
            computation_hash=computation_hash,
            model_version=self.model_version,
        )

    def explain_hotspot(
        self,
        sensor_readings: Dict[str, float],
        hotspot_severity: float,
        prediction_fn: Optional[Callable] = None,
        tube_id: Optional[str] = None,
    ) -> LIMEResult:
        """
        Explain a hotspot detection prediction.

        Args:
            sensor_readings: Current sensor values
            hotspot_severity: Severity score (0-100)
            prediction_fn: Optional custom prediction function
            tube_id: Optional tube identifier

        Returns:
            LIMEResult explaining the hotspot detection
        """
        if prediction_fn is None:
            prediction_fn = self._create_hotspot_predictor(sensor_readings)

        prediction_id = f"hotspot_{tube_id}" if tube_id else self._generate_id()

        return self.explain_prediction(
            sensor_readings=sensor_readings,
            predicted_value=hotspot_severity,
            prediction_fn=prediction_fn,
            prediction_type="hotspot",
            prediction_id=prediction_id,
        )

    def explain_efficiency(
        self,
        sensor_readings: Dict[str, float],
        efficiency_value: float,
        prediction_fn: Optional[Callable] = None,
    ) -> LIMEResult:
        """
        Explain an efficiency prediction.

        Args:
            sensor_readings: Current sensor values
            efficiency_value: Efficiency percentage
            prediction_fn: Optional custom prediction function

        Returns:
            LIMEResult explaining the efficiency prediction
        """
        if prediction_fn is None:
            prediction_fn = self._create_efficiency_predictor(sensor_readings)

        return self.explain_prediction(
            sensor_readings=sensor_readings,
            predicted_value=efficiency_value,
            prediction_fn=prediction_fn,
            prediction_type="efficiency",
        )

    def explain_rul(
        self,
        sensor_readings: Dict[str, float],
        rul_hours: float,
        prediction_fn: Optional[Callable] = None,
        component_id: str = "furnace_tube",
    ) -> LIMEResult:
        """
        Explain a Remaining Useful Life prediction.

        Args:
            sensor_readings: Current sensor values
            rul_hours: Predicted remaining hours
            prediction_fn: Optional custom prediction function
            component_id: Component being assessed

        Returns:
            LIMEResult explaining the RUL prediction
        """
        if prediction_fn is None:
            prediction_fn = self._create_rul_predictor(sensor_readings)

        return self.explain_prediction(
            sensor_readings=sensor_readings,
            predicted_value=rul_hours,
            prediction_fn=prediction_fn,
            prediction_type="rul",
            prediction_id=f"rul_{component_id}",
        )

    def compare_with_shap(
        self,
        lime_result: LIMEResult,
        shap_importance: Dict[str, float],
        top_k: int = 5,
    ) -> ComparisonResult:
        """
        Compare LIME explanation with SHAP explanation.

        Args:
            lime_result: LIME explanation result
            shap_importance: SHAP feature importance dictionary
            top_k: Number of top features to compare

        Returns:
            ComparisonResult with agreement analysis
        """
        # Get top features from each method
        lime_top = sorted(
            lime_result.feature_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]
        lime_top_names = [name for name, _ in lime_top]

        shap_top = sorted(
            shap_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]
        shap_top_names = [name for name, _ in shap_top]

        # Calculate overlap
        overlap = set(lime_top_names) & set(shap_top_names)
        agreement_score = len(overlap) / top_k

        # Find disagreements
        disagreements = []
        for name in set(lime_top_names) - overlap:
            if name in shap_importance:
                disagreements.append({
                    "feature": name,
                    "lime_rank": lime_top_names.index(name) + 1,
                    "shap_importance": shap_importance[name],
                    "reason": "High LIME importance, lower SHAP importance",
                })

        for name in set(shap_top_names) - overlap:
            if name in lime_result.feature_weights:
                disagreements.append({
                    "feature": name,
                    "shap_rank": shap_top_names.index(name) + 1,
                    "lime_weight": lime_result.feature_weights[name],
                    "reason": "High SHAP importance, lower LIME importance",
                })

        # Generate recommendation
        if agreement_score >= 0.8:
            recommendation = "Strong agreement between methods. High confidence in explanation."
        elif agreement_score >= 0.5:
            recommendation = "Moderate agreement. Consider both explanations for full picture."
        else:
            recommendation = "Low agreement. May indicate non-linear interactions or model complexity."

        # Combined confidence
        combined_confidence = (
            agreement_score * 0.5 +
            (lime_result.score * 0.3) +
            (0.2 if lime_result.confidence_level == ConfidenceLevel.HIGH else 0.1)
        )

        return ComparisonResult(
            agreement_score=round(agreement_score, 4),
            top_features_overlap=list(overlap),
            disagreements=disagreements,
            recommendation=recommendation,
            combined_confidence=round(combined_confidence, 4),
        )

    def get_confidence_assessment(
        self,
        result: LIMEResult,
    ) -> Dict[str, Any]:
        """
        Get detailed confidence assessment for explanation.

        Args:
            result: LIME explanation result

        Returns:
            Dictionary with confidence metrics and interpretation
        """
        r2_score = result.score

        assessment = {
            "r2_score": r2_score,
            "confidence_level": result.confidence_level.value,
            "num_features": result.num_features_used,
            "interpretation": "",
            "recommendations": [],
        }

        if r2_score >= 0.8:
            assessment["interpretation"] = (
                "The local linear model explains the prediction well. "
                "The explanation is reliable and can be used for decision making."
            )
        elif r2_score >= 0.5:
            assessment["interpretation"] = (
                "The local linear model provides a reasonable approximation. "
                "Some non-linear effects may not be fully captured."
            )
            assessment["recommendations"].append(
                "Consider cross-validating with SHAP explanations"
            )
        else:
            assessment["interpretation"] = (
                "The local linear model has limited explanatory power. "
                "The prediction may involve complex non-linear interactions."
            )
            assessment["recommendations"].extend([
                "Use SHAP as primary explanation method",
                "Consider ensemble of local explanations",
                "Investigate feature interactions",
            ])

        # Check for dominant features
        weights = list(result.feature_weights.values())
        if weights:
            max_weight = max(abs(w) for w in weights)
            total_weight = sum(abs(w) for w in weights)
            dominance = max_weight / total_weight if total_weight > 0 else 0

            if dominance > 0.7:
                assessment["recommendations"].append(
                    f"Single feature dominates explanation ({dominance:.0%}). "
                    "Verify this matches domain knowledge."
                )

        return assessment

    def _extract_features(
        self,
        sensor_readings: Dict[str, float],
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract numerical features from sensor readings."""
        sorted_keys = sorted(sensor_readings.keys())
        feature_names = sorted_keys
        features = np.array([sensor_readings[k] for k in sorted_keys])
        return features, feature_names

    def _generate_training_data(
        self,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Generate synthetic training data around reference point."""
        np.random.seed(self.random_state)
        n_samples = self.num_samples
        n_features = len(reference)

        # Generate variations with reasonable bounds
        variations = np.random.randn(n_samples, n_features) * 0.2
        training_data = reference + variations * np.abs(reference + 1e-6)

        # Ensure positive values for sensor readings
        training_data = np.maximum(training_data, 0.01)

        return training_data

    def _calculate_lime_explanation(
        self,
        features: np.ndarray,
        feature_names: List[str],
        prediction_fn: Callable,
    ) -> Dict[str, Any]:
        """Calculate LIME explanation using lime library."""
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=self._training_data,
            feature_names=feature_names,
            mode="regression",
            discretize_continuous=self.discretize_continuous,
            kernel_width=self.kernel_width * np.sqrt(len(features)),
            random_state=self.random_state,
        )

        # Get explanation
        exp = explainer.explain_instance(
            features,
            prediction_fn,
            num_features=self.num_features,
            num_samples=self.num_samples,
        )

        # Extract weights
        feature_weights = {}
        for feat, weight in exp.as_list():
            # LIME returns discretized feature names; map back to original
            original_name = self._map_to_original_feature(feat, feature_names)
            feature_weights[original_name] = round(weight, 6)

        local_prediction = prediction_fn(features.reshape(1, -1))[0]
        intercept = float(exp.intercept[0]) if hasattr(exp, 'intercept') else 0.0
        score = float(exp.score) if hasattr(exp, 'score') else 0.0

        return {
            "weights": feature_weights,
            "local_prediction": float(local_prediction),
            "intercept": intercept,
            "score": score,
        }

    def _calculate_fallback_explanation(
        self,
        features: np.ndarray,
        feature_names: List[str],
        prediction_fn: Callable,
    ) -> Dict[str, Any]:
        """Fallback weighted linear regression when LIME not available."""
        np.random.seed(self.random_state)
        n_samples = self.num_samples

        # Generate perturbed samples
        perturbations = np.random.randn(n_samples, len(features)) * 0.1
        X_perturbed = features + perturbations * np.abs(features + 1e-6)
        X_perturbed = np.maximum(X_perturbed, 0.01)

        # Get predictions
        y = prediction_fn(X_perturbed)

        # Compute weights (exponential kernel)
        distances = np.sqrt(np.sum(perturbations ** 2, axis=1))
        weights = np.exp(-distances ** 2 / (2 * self.kernel_width ** 2))

        # Weighted linear regression
        try:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0, random_state=self.random_state)
            model.fit(X_perturbed, y, sample_weight=weights)
            coefficients = model.coef_
            intercept = float(model.intercept_)
            score = float(model.score(X_perturbed, y, sample_weight=weights))
        except ImportError:
            # Manual weighted least squares
            W = np.diag(weights)
            X_aug = np.column_stack([np.ones(n_samples), X_perturbed])
            try:
                beta = np.linalg.solve(
                    X_aug.T @ W @ X_aug + np.eye(X_aug.shape[1]) * 0.01,
                    X_aug.T @ W @ y
                )
                intercept = float(beta[0])
                coefficients = beta[1:]
                y_pred = X_aug @ beta
                ss_res = float(np.sum(weights * (y - y_pred) ** 2))
                ss_tot = float(np.sum(weights * (y - np.average(y, weights=weights)) ** 2))
                score = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            except np.linalg.LinAlgError:
                coefficients = np.zeros(len(features))
                intercept = 0.0
                score = 0.0

        # Build feature weights
        feature_weights = {}
        indices = np.argsort(np.abs(coefficients))[::-1][:self.num_features]

        for i in indices:
            feature_weights[feature_names[i]] = round(float(coefficients[i]), 6)

        local_prediction = float(prediction_fn(features.reshape(1, -1))[0])

        return {
            "weights": feature_weights,
            "local_prediction": local_prediction,
            "intercept": intercept,
            "score": score,
        }

    def _determine_confidence(self, score: float) -> ConfidenceLevel:
        """Determine confidence level from R2 score."""
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _generate_tabular_explanations(
        self,
        weights: Dict[str, float],
        sensor_readings: Dict[str, float],
    ) -> List[TabularExplanation]:
        """Generate structured tabular explanations."""
        explanations = []

        sorted_weights = sorted(
            weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for name, weight in sorted_weights:
            value = sensor_readings.get(name, 0.0)
            direction = "positive" if weight > 0 else "negative"

            display_name = self._get_display_name(name)
            unit = self._get_unit(name)
            value_str = f"{value:.2f} {unit}".strip()

            if weight > 0:
                contribution_text = f"{display_name} at {value_str} increases prediction"
            else:
                contribution_text = f"{display_name} at {value_str} decreases prediction"

            explanations.append(TabularExplanation(
                feature_name=name,
                feature_value=value,
                weight=weight,
                direction=direction,
                contribution_text=contribution_text,
            ))

        return explanations

    def _generate_explanation_text(
        self,
        weights: Dict[str, float],
        sensor_readings: Dict[str, float],
        prediction_type: str,
    ) -> List[str]:
        """Generate human-readable explanation text."""
        explanation_lines = []

        # Header
        type_descriptions = {
            "hotspot": "hotspot severity",
            "efficiency": "furnace efficiency",
            "rul": "remaining useful life",
        }
        type_desc = type_descriptions.get(prediction_type, prediction_type)

        sorted_weights = sorted(
            weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for name, weight in sorted_weights[:self.num_features]:
            value = sensor_readings.get(name, 0.0)
            display_name = self._get_display_name(name)
            unit = self._get_unit(name)

            direction = "increases" if weight > 0 else "decreases"
            abs_weight = abs(weight)

            line = (
                f"{display_name} = {value:.2f}{unit}: "
                f"{direction} {type_desc} by {abs_weight:.4f}"
            )
            explanation_lines.append(line)

        return explanation_lines

    def _get_display_name(self, feature_name: str) -> str:
        """Get human-readable display name for feature."""
        name_lower = feature_name.lower()

        for pattern, display in self.FEATURE_DISPLAY_NAMES.items():
            if pattern in name_lower:
                # Add any suffix numbers
                suffix = ''.join(c for c in feature_name if c.isdigit())
                if suffix:
                    return f"{display} {suffix}"
                return display

        # Fallback: capitalize and replace underscores
        return feature_name.replace("_", " ").title()

    def _get_unit(self, feature_name: str) -> str:
        """Get unit for feature."""
        name_lower = feature_name.lower()

        for pattern, unit in self.FEATURE_UNITS.items():
            if pattern in name_lower:
                return unit

        return ""

    def _map_to_original_feature(
        self,
        lime_feature: str,
        feature_names: List[str],
    ) -> str:
        """Map LIME discretized feature name to original."""
        # LIME may return names like "feature <= 0.5" or "0.3 < feature <= 0.7"
        for name in feature_names:
            if name in lime_feature:
                return name

        # If no match, return as-is (might already be original)
        return lime_feature.split()[0] if " " in lime_feature else lime_feature

    def _create_hotspot_predictor(
        self,
        reference_readings: Dict[str, float],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create default hotspot prediction function."""
        feature_names = sorted(reference_readings.keys())

        def predictor(X: np.ndarray) -> np.ndarray:
            predictions = []
            for row in X:
                # Hotspot severity driven by temperature features
                temp_indices = [
                    i for i, name in enumerate(feature_names)
                    if "tmt" in name.lower() or "temp" in name.lower()
                ]

                if temp_indices:
                    temp_values = row[temp_indices]
                    # Higher temperatures = higher severity
                    max_temp = np.max(temp_values)
                    severity = min(100, max(0, (max_temp - 400) / 2))
                else:
                    severity = np.mean(row) * 0.1

                predictions.append(severity)

            return np.array(predictions)

        return predictor

    def _create_efficiency_predictor(
        self,
        reference_readings: Dict[str, float],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create default efficiency prediction function."""
        def predictor(X: np.ndarray) -> np.ndarray:
            predictions = []
            for row in X:
                # Efficiency based on variance (lower variance = better efficiency)
                variance = np.var(row)
                efficiency = max(50, min(100, 95 - variance * 0.1))
                predictions.append(efficiency)

            return np.array(predictions)

        return predictor

    def _create_rul_predictor(
        self,
        reference_readings: Dict[str, float],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create default RUL prediction function."""
        feature_names = sorted(reference_readings.keys())

        def predictor(X: np.ndarray) -> np.ndarray:
            predictions = []
            for row in X:
                # RUL decreases with high temperatures
                temp_indices = [
                    i for i, name in enumerate(feature_names)
                    if "tmt" in name.lower() or "temp" in name.lower()
                ]

                if temp_indices:
                    max_temp = np.max(row[temp_indices])
                    # Higher temp = lower RUL
                    rul = max(100, 10000 - max_temp * 10)
                else:
                    rul = 5000 - np.max(row) * 5

                predictions.append(max(0, rul))

            return np.array(predictions)

        return predictor

    def _generate_id(self) -> str:
        """Generate unique prediction ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]

    def _compute_hash(
        self,
        features: np.ndarray,
        weights: Dict[str, float],
        predicted_value: float,
    ) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        data = {
            "features": features.tolist(),
            "weights": weights,
            "predicted_value": predicted_value,
            "model_version": self.model_version,
            "explainer_version": self.VERSION,
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
