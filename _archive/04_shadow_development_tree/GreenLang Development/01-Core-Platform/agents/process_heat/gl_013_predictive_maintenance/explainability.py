# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Explainability Module

This module implements Local Interpretable Model-agnostic Explanations (LIME)
and SHAP (SHapley Additive exPlanations) for predictive maintenance decisions.

LIME provides local explanations by perturbing input features and observing
how predictions change. This helps maintenance engineers understand WHY
a particular failure prediction was made.

SHAP provides global and local feature importance using game-theoretic
Shapley values, offering consistent and theoretically grounded explanations.

Key Features:
- LIME local explanations for individual predictions
- SHAP global feature importance analysis
- Counterfactual explanations ("what-if" scenarios)
- Feature contribution visualization support
- Deterministic explanation generation (ZERO HALLUCINATION)
- Provenance tracking for audit compliance

All explanations are DETERMINISTIC and REPRODUCIBLE for regulatory compliance.

Example:
    >>> from greenlang.agents.process_heat.gl_013_predictive_maintenance.explainability import (
    ...     LIMEExplainer, SHAPAnalyzer
    ... )
    >>> explainer = LIMEExplainer(prediction_engine)
    >>> explanation = explainer.explain(features, prediction)
    >>> print(explanation.summary)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import math
import random

from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    FailureMode,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.schemas import (
    FailurePrediction,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

FeatureVector = Dict[str, float]
PredictionFunction = Callable[[FeatureVector], float]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeatureContribution:
    """Contribution of a single feature to the prediction."""
    feature_name: str
    feature_value: float
    contribution: float
    direction: str  # "positive" or "negative"
    importance_rank: int
    human_readable_name: str
    unit: str = ""


@dataclass
class LIMEExplanation:
    """LIME explanation for a single prediction."""
    prediction_id: str
    failure_mode: FailureMode
    base_probability: float
    explained_probability: float
    intercept: float
    feature_contributions: List[FeatureContribution]
    r_squared: float  # Local model fit quality
    num_samples: int
    kernel_width: float
    summary: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""


@dataclass
class SHAPValues:
    """SHAP values for feature importance."""
    feature_name: str
    shap_value: float
    base_value: float
    feature_value: float
    contribution_pct: float


@dataclass
class SHAPExplanation:
    """SHAP explanation for predictions."""
    prediction_id: str
    failure_mode: FailureMode
    base_value: float  # Expected value (average prediction)
    predicted_value: float
    shap_values: List[SHAPValues]
    total_contribution: float
    main_effects: Dict[str, float]
    interaction_effects: Dict[Tuple[str, str], float]
    summary: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""


@dataclass
class CounterfactualExplanation:
    """Counterfactual explanation showing what would change the prediction."""
    prediction_id: str
    failure_mode: FailureMode
    original_probability: float
    target_probability: float
    feature_changes: Dict[str, Tuple[float, float]]  # feature -> (old, new)
    distance: float  # How different is the counterfactual
    feasibility_score: float  # How feasible are the changes
    action_recommendations: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class GlobalExplanation:
    """Global model explanation across all failure modes."""
    model_id: str
    feature_importance: Dict[str, float]  # Overall importance
    feature_importance_by_mode: Dict[FailureMode, Dict[str, float]]
    feature_interactions: List[Tuple[str, str, float]]  # Top interactions
    monotonicity: Dict[str, str]  # Feature monotonicity direction
    summary: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# LIME EXPLAINER
# =============================================================================

class LIMEExplainer:
    """
    Local Interpretable Model-agnostic Explanations (LIME) for PdM predictions.

    LIME explains individual predictions by:
    1. Generating perturbed samples around the instance
    2. Getting predictions for perturbed samples
    3. Fitting a simple linear model weighted by proximity
    4. Using linear model coefficients as feature importance

    This implementation is DETERMINISTIC for reproducibility.

    Attributes:
        prediction_fn: Function that returns probability for feature vector
        feature_names: List of feature names
        num_samples: Number of perturbed samples to generate
        kernel_width: Width of exponential kernel for weighting

    Example:
        >>> explainer = LIMEExplainer(engine.predict_probability)
        >>> explanation = explainer.explain(features, FailureMode.BEARING_WEAR)
        >>> for contrib in explanation.feature_contributions[:5]:
        ...     print(f"{contrib.human_readable_name}: {contrib.contribution:.3f}")
    """

    # Human-readable feature names
    FEATURE_DISPLAY_NAMES = {
        "velocity_rms_normalized": "Vibration Level",
        "acceleration_rms_normalized": "Acceleration Level",
        "bearing_defect_indicator": "Bearing Defect Signal",
        "imbalance_indicator": "Rotor Imbalance Signal",
        "misalignment_indicator": "Shaft Misalignment Signal",
        "viscosity_change_pct": "Oil Viscosity Change",
        "tan_normalized": "Oil Acidity (TAN)",
        "iron_ppm_normalized": "Wear Metal (Iron)",
        "water_ppm_normalized": "Water Contamination",
        "particle_count_score": "Particle Count",
        "temperature_normalized": "Operating Temperature",
        "delta_t_normalized": "Temperature Differential",
        "rotor_bar_severity_db": "Rotor Bar Condition",
        "eccentricity_severity_db": "Rotor Eccentricity",
        "current_unbalance_pct": "Current Unbalance",
        "running_hours_normalized": "Equipment Age",
        "load_factor": "Load Factor",
        "health_score_composite": "Overall Health Score",
    }

    FEATURE_UNITS = {
        "velocity_rms_normalized": "ratio",
        "viscosity_change_pct": "%",
        "temperature_normalized": "ratio",
        "current_unbalance_pct": "%",
        "load_factor": "ratio",
        "running_hours_normalized": "ratio",
    }

    def __init__(
        self,
        prediction_fn: PredictionFunction,
        feature_names: Optional[List[str]] = None,
        num_samples: int = 500,
        kernel_width: float = 0.75,
        random_seed: int = 42,
    ) -> None:
        """
        Initialize LIME explainer.

        Args:
            prediction_fn: Function that takes features and returns probability
            feature_names: List of feature names (optional, detected from input)
            num_samples: Number of perturbed samples for local model
            kernel_width: Width of exponential kernel (controls locality)
            random_seed: Random seed for reproducibility
        """
        self.prediction_fn = prediction_fn
        self.feature_names = feature_names or []
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.random_seed = random_seed

        # Initialize deterministic random generator
        self._rng = random.Random(random_seed)

        logger.info(
            f"LIMEExplainer initialized: samples={num_samples}, "
            f"kernel_width={kernel_width}"
        )

    def explain(
        self,
        features: FeatureVector,
        failure_mode: FailureMode,
        prediction: Optional[FailurePrediction] = None,
    ) -> LIMEExplanation:
        """
        Generate LIME explanation for a prediction.

        Args:
            features: Feature vector for the instance
            failure_mode: Failure mode being explained
            prediction: Optional pre-computed prediction

        Returns:
            LIMEExplanation with feature contributions
        """
        logger.info(f"Generating LIME explanation for {failure_mode.value}")

        # Get base prediction
        base_prob = self.prediction_fn(features)
        if prediction is not None:
            base_prob = prediction.probability

        # Update feature names from input if not set
        if not self.feature_names:
            self.feature_names = list(features.keys())

        # Generate perturbed samples
        perturbed_samples, perturbed_features = self._generate_perturbations(
            features
        )

        # Get predictions for perturbed samples
        perturbed_predictions = [
            self.prediction_fn(sample) for sample in perturbed_samples
        ]

        # Calculate distances and weights
        distances = self._calculate_distances(features, perturbed_samples)
        weights = self._calculate_kernel_weights(distances)

        # Fit weighted linear model
        coefficients, intercept, r_squared = self._fit_linear_model(
            perturbed_features,
            perturbed_predictions,
            weights,
        )

        # Create feature contributions
        contributions = self._create_contributions(
            features,
            coefficients,
        )

        # Calculate explained probability
        explained_prob = intercept + sum(
            coef * features.get(name, 0)
            for name, coef in coefficients.items()
        )
        explained_prob = max(0, min(1, explained_prob))

        # Generate summary
        summary = self._generate_summary(
            failure_mode,
            base_prob,
            contributions[:5],
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance(
            features,
            failure_mode,
            coefficients,
        )

        prediction_id = f"LIME-{failure_mode.value}-{datetime.now(timezone.utc).isoformat()}"

        return LIMEExplanation(
            prediction_id=prediction_id,
            failure_mode=failure_mode,
            base_probability=base_prob,
            explained_probability=explained_prob,
            intercept=intercept,
            feature_contributions=contributions,
            r_squared=r_squared,
            num_samples=self.num_samples,
            kernel_width=self.kernel_width,
            summary=summary,
            provenance_hash=provenance_hash,
        )

    def _generate_perturbations(
        self,
        features: FeatureVector,
    ) -> Tuple[List[FeatureVector], List[List[float]]]:
        """
        Generate perturbed samples around the instance.

        Uses deterministic Gaussian perturbations for reproducibility.

        Args:
            features: Original feature vector

        Returns:
            Tuple of (perturbed samples, feature matrix)
        """
        samples = []
        feature_matrix = []

        for i in range(self.num_samples):
            # Deterministic perturbation based on sample index
            perturbed = {}
            feature_row = []

            for name in self.feature_names:
                original_value = features.get(name, 0)

                # Deterministic "random" perturbation
                # Use hash of (seed, sample_index, feature_name) for reproducibility
                hash_input = f"{self.random_seed}_{i}_{name}"
                hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
                normalized = (hash_val / 0xFFFFFFFF) * 2 - 1  # Map to [-1, 1]

                # Apply perturbation with some probability
                perturbation_prob = (hash_val % 100) / 100
                if perturbation_prob < 0.7:  # Perturb 70% of samples
                    perturbation = normalized * 0.3  # Scale perturbation
                    new_value = original_value + perturbation
                    # Clip to valid range
                    new_value = max(0, min(1, new_value))
                else:
                    new_value = original_value

                perturbed[name] = new_value
                feature_row.append(new_value)

            samples.append(perturbed)
            feature_matrix.append(feature_row)

        return samples, feature_matrix

    def _calculate_distances(
        self,
        original: FeatureVector,
        samples: List[FeatureVector],
    ) -> List[float]:
        """Calculate Euclidean distances from original to perturbed samples."""
        distances = []

        for sample in samples:
            distance_sq = sum(
                (original.get(name, 0) - sample.get(name, 0)) ** 2
                for name in self.feature_names
            )
            distances.append(math.sqrt(distance_sq))

        return distances

    def _calculate_kernel_weights(
        self,
        distances: List[float],
    ) -> List[float]:
        """Calculate exponential kernel weights."""
        weights = []

        for d in distances:
            # Exponential kernel: exp(-d^2 / kernel_width^2)
            weight = math.exp(-(d ** 2) / (self.kernel_width ** 2))
            weights.append(weight)

        return weights

    def _fit_linear_model(
        self,
        feature_matrix: List[List[float]],
        predictions: List[float],
        weights: List[float],
    ) -> Tuple[Dict[str, float], float, float]:
        """
        Fit weighted linear regression model.

        Uses analytical solution for weighted least squares.

        Args:
            feature_matrix: Matrix of perturbed features
            predictions: Predictions for perturbed samples
            weights: Sample weights

        Returns:
            Tuple of (coefficients, intercept, R-squared)
        """
        n = len(predictions)
        p = len(self.feature_names)

        if n == 0 or p == 0:
            return {name: 0.0 for name in self.feature_names}, 0.5, 0.0

        # Calculate weighted means
        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1

        y_mean = sum(w * y for w, y in zip(weights, predictions)) / total_weight
        x_means = [
            sum(w * row[j] for w, row in zip(weights, feature_matrix)) / total_weight
            for j in range(p)
        ]

        # Simple weighted linear regression (feature-by-feature)
        coefficients = {}

        for j, name in enumerate(self.feature_names):
            # Calculate weighted covariance and variance
            cov_xy = sum(
                weights[i] * (feature_matrix[i][j] - x_means[j]) * (predictions[i] - y_mean)
                for i in range(n)
            )
            var_x = sum(
                weights[i] * (feature_matrix[i][j] - x_means[j]) ** 2
                for i in range(n)
            )

            if var_x > 1e-10:
                coefficients[name] = cov_xy / var_x
            else:
                coefficients[name] = 0.0

        # Calculate intercept
        intercept = y_mean - sum(
            coefficients.get(name, 0) * x_means[j]
            for j, name in enumerate(self.feature_names)
        )

        # Calculate R-squared
        ss_tot = sum(
            weights[i] * (predictions[i] - y_mean) ** 2
            for i in range(n)
        )

        ss_res = 0
        for i in range(n):
            pred = intercept + sum(
                coefficients.get(name, 0) * feature_matrix[i][j]
                for j, name in enumerate(self.feature_names)
            )
            ss_res += weights[i] * (predictions[i] - pred) ** 2

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r_squared = max(0, min(1, r_squared))

        return coefficients, intercept, r_squared

    def _create_contributions(
        self,
        features: FeatureVector,
        coefficients: Dict[str, float],
    ) -> List[FeatureContribution]:
        """Create sorted list of feature contributions."""
        contributions = []

        # Calculate contributions
        feature_contribs = []
        for name, coef in coefficients.items():
            value = features.get(name, 0)
            contribution = coef * value
            feature_contribs.append((name, value, coef, contribution))

        # Sort by absolute contribution
        feature_contribs.sort(key=lambda x: abs(x[3]), reverse=True)

        # Create contribution objects
        for rank, (name, value, coef, contribution) in enumerate(feature_contribs, 1):
            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=value,
                contribution=contribution,
                direction="positive" if contribution > 0 else "negative",
                importance_rank=rank,
                human_readable_name=self.FEATURE_DISPLAY_NAMES.get(
                    name, name.replace("_", " ").title()
                ),
                unit=self.FEATURE_UNITS.get(name, ""),
            ))

        return contributions

    def _generate_summary(
        self,
        failure_mode: FailureMode,
        probability: float,
        top_contributions: List[FeatureContribution],
    ) -> str:
        """Generate human-readable summary of the explanation."""
        mode_name = failure_mode.value.replace("_", " ").title()

        # Categorize probability
        if probability > 0.7:
            risk_level = "high"
        elif probability > 0.4:
            risk_level = "moderate"
        else:
            risk_level = "low"

        summary_lines = [
            f"The {mode_name} failure prediction has {risk_level} probability ({probability:.1%}).",
            "",
            "Key factors contributing to this prediction:",
        ]

        for contrib in top_contributions:
            direction = "increases" if contrib.direction == "positive" else "decreases"
            summary_lines.append(
                f"  - {contrib.human_readable_name}: {direction} risk "
                f"(contribution: {contrib.contribution:+.3f})"
            )

        return "\n".join(summary_lines)

    def _calculate_provenance(
        self,
        features: FeatureVector,
        failure_mode: FailureMode,
        coefficients: Dict[str, float],
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        feature_str = "|".join(
            f"{k}:{v:.6f}" for k, v in sorted(features.items())
        )
        coef_str = "|".join(
            f"{k}:{v:.6f}" for k, v in sorted(coefficients.items())
        )
        provenance_str = (
            f"LIME|{failure_mode.value}|{feature_str}|{coef_str}|"
            f"{self.num_samples}|{self.kernel_width}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# SHAP ANALYZER
# =============================================================================

class SHAPAnalyzer:
    """
    SHAP (SHapley Additive exPlanations) analyzer for PdM predictions.

    SHAP values represent the contribution of each feature to the difference
    between the prediction and the average prediction. They have desirable
    properties:

    - Local accuracy: Contributions sum to the prediction difference
    - Consistency: If a feature's contribution increases, its SHAP value increases
    - Missingness: Features with no influence have SHAP value of 0

    This implementation uses a simplified Kernel SHAP approximation
    that is DETERMINISTIC for reproducibility.

    Attributes:
        prediction_fn: Function that returns probability for feature vector
        feature_names: List of feature names
        background_data: Background dataset for computing expected values

    Example:
        >>> analyzer = SHAPAnalyzer(engine.predict_probability, background_samples)
        >>> explanation = analyzer.explain(features, FailureMode.BEARING_WEAR)
        >>> for sv in explanation.shap_values[:5]:
        ...     print(f"{sv.feature_name}: {sv.shap_value:.3f}")
    """

    def __init__(
        self,
        prediction_fn: PredictionFunction,
        background_data: Optional[List[FeatureVector]] = None,
        feature_names: Optional[List[str]] = None,
        num_background_samples: int = 100,
    ) -> None:
        """
        Initialize SHAP analyzer.

        Args:
            prediction_fn: Function that takes features and returns probability
            background_data: Background samples for expected value calculation
            feature_names: List of feature names
            num_background_samples: Number of background samples to use
        """
        self.prediction_fn = prediction_fn
        self.feature_names = feature_names or []
        self.num_background_samples = num_background_samples

        # Store or generate background data
        if background_data:
            self.background_data = background_data[:num_background_samples]
        else:
            self.background_data = self._generate_default_background()

        # Compute expected value (base value)
        self.expected_value = self._compute_expected_value()

        logger.info(
            f"SHAPAnalyzer initialized: background_samples={len(self.background_data)}, "
            f"expected_value={self.expected_value:.4f}"
        )

    def _generate_default_background(self) -> List[FeatureVector]:
        """Generate default background data for SHAP."""
        # Create synthetic background with typical feature values
        default_features = {
            "velocity_rms_normalized": 0.2,
            "acceleration_rms_normalized": 0.15,
            "bearing_defect_indicator": 0.0,
            "imbalance_indicator": 0.0,
            "misalignment_indicator": 0.0,
            "viscosity_change_pct": 0.05,
            "tan_normalized": 0.15,
            "iron_ppm_normalized": 0.1,
            "water_ppm_normalized": 0.08,
            "temperature_normalized": 0.35,
            "delta_t_normalized": 0.2,
            "rotor_bar_severity_db": 0.0,
            "eccentricity_severity_db": 0.0,
            "current_unbalance_pct": 0.05,
            "running_hours_normalized": 0.4,
            "load_factor": 0.8,
            "health_score_composite": 85.0,
        }

        # Generate variations
        background = []
        for i in range(self.num_background_samples):
            sample = {}
            for name, base_value in default_features.items():
                # Deterministic variation
                hash_input = f"background_{i}_{name}"
                hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
                variation = (hash_val / 0xFFFFFFFF - 0.5) * 0.4

                new_value = base_value + variation * base_value
                new_value = max(0, min(1, new_value))
                sample[name] = new_value

            background.append(sample)

        return background

    def _compute_expected_value(self) -> float:
        """Compute expected (average) prediction over background data."""
        if not self.background_data:
            return 0.5

        predictions = [
            self.prediction_fn(sample)
            for sample in self.background_data
        ]

        return sum(predictions) / len(predictions)

    def explain(
        self,
        features: FeatureVector,
        failure_mode: FailureMode,
        prediction: Optional[FailurePrediction] = None,
    ) -> SHAPExplanation:
        """
        Generate SHAP explanation for a prediction.

        Uses Kernel SHAP approximation for model-agnostic explanations.

        Args:
            features: Feature vector for the instance
            failure_mode: Failure mode being explained
            prediction: Optional pre-computed prediction

        Returns:
            SHAPExplanation with SHAP values
        """
        logger.info(f"Generating SHAP explanation for {failure_mode.value}")

        # Update feature names from input
        if not self.feature_names:
            self.feature_names = list(features.keys())

        # Get actual prediction
        predicted_value = self.prediction_fn(features)
        if prediction is not None:
            predicted_value = prediction.probability

        # Calculate SHAP values using simplified Kernel SHAP
        shap_values = self._calculate_shap_values(features)

        # Create SHAP value objects
        shap_value_list = []
        total_contribution = 0

        for name in sorted(shap_values.keys(), key=lambda x: abs(shap_values[x]), reverse=True):
            sv = shap_values[name]
            total_contribution += sv

            contribution_pct = (sv / (predicted_value - self.expected_value) * 100
                               if abs(predicted_value - self.expected_value) > 0.001
                               else 0)

            shap_value_list.append(SHAPValues(
                feature_name=name,
                shap_value=sv,
                base_value=self.expected_value,
                feature_value=features.get(name, 0),
                contribution_pct=contribution_pct,
            ))

        # Calculate main effects (individual feature contributions)
        main_effects = {name: sv for name, sv in shap_values.items()}

        # Calculate interaction effects (simplified - top pairs)
        interaction_effects = self._calculate_interactions(features, shap_values)

        # Generate summary
        summary = self._generate_summary(
            failure_mode,
            predicted_value,
            shap_value_list[:5],
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            features,
            failure_mode,
            shap_values,
        )

        prediction_id = f"SHAP-{failure_mode.value}-{datetime.now(timezone.utc).isoformat()}"

        return SHAPExplanation(
            prediction_id=prediction_id,
            failure_mode=failure_mode,
            base_value=self.expected_value,
            predicted_value=predicted_value,
            shap_values=shap_value_list,
            total_contribution=total_contribution,
            main_effects=main_effects,
            interaction_effects=interaction_effects,
            summary=summary,
            provenance_hash=provenance_hash,
        )

    def _calculate_shap_values(
        self,
        features: FeatureVector,
    ) -> Dict[str, float]:
        """
        Calculate SHAP values using simplified Kernel SHAP.

        This uses permutation-based approximation for efficiency.
        """
        shap_values = {name: 0.0 for name in self.feature_names}

        # For each feature, estimate its marginal contribution
        for target_feature in self.feature_names:
            contribution = 0.0

            # Sample coalitions (subsets of features)
            num_coalitions = 50  # Reduced for efficiency
            for i in range(num_coalitions):
                # Deterministic coalition selection
                hash_input = f"coalition_{target_feature}_{i}"
                hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)

                # Randomly include features in coalition
                coalition = set()
                for j, name in enumerate(self.feature_names):
                    if name == target_feature:
                        continue
                    feature_hash = int(hashlib.md5(
                        f"{hash_input}_{j}".encode()
                    ).hexdigest()[:4], 16)
                    if feature_hash % 2 == 0:
                        coalition.add(name)

                # Prediction with target feature
                features_with = self._create_masked_features(
                    features, coalition | {target_feature}
                )
                pred_with = self.prediction_fn(features_with)

                # Prediction without target feature
                features_without = self._create_masked_features(
                    features, coalition
                )
                pred_without = self.prediction_fn(features_without)

                # Marginal contribution
                contribution += pred_with - pred_without

            shap_values[target_feature] = contribution / num_coalitions

        return shap_values

    def _create_masked_features(
        self,
        features: FeatureVector,
        include_features: set,
    ) -> FeatureVector:
        """Create feature vector with only specified features."""
        result = {}
        for name in self.feature_names:
            if name in include_features:
                result[name] = features.get(name, 0)
            else:
                # Use average from background
                bg_values = [
                    sample.get(name, 0) for sample in self.background_data
                ]
                result[name] = sum(bg_values) / len(bg_values) if bg_values else 0

        return result

    def _calculate_interactions(
        self,
        features: FeatureVector,
        shap_values: Dict[str, float],
    ) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise feature interactions (simplified)."""
        interactions = {}

        # Consider only top features for interactions
        top_features = sorted(
            shap_values.keys(),
            key=lambda x: abs(shap_values[x]),
            reverse=True
        )[:5]

        for i, f1 in enumerate(top_features):
            for f2 in top_features[i + 1:]:
                # Simplified interaction: product of contributions
                interaction = shap_values[f1] * shap_values[f2]
                if abs(interaction) > 0.001:
                    interactions[(f1, f2)] = interaction

        return interactions

    def _generate_summary(
        self,
        failure_mode: FailureMode,
        predicted_value: float,
        top_shap_values: List[SHAPValues],
    ) -> str:
        """Generate human-readable SHAP summary."""
        mode_name = failure_mode.value.replace("_", " ").title()
        diff = predicted_value - self.expected_value

        if diff > 0:
            direction = "above"
        else:
            direction = "below"

        summary_lines = [
            f"SHAP Analysis for {mode_name}:",
            f"Predicted probability: {predicted_value:.1%} "
            f"({abs(diff):.1%} {direction} average)",
            "",
            "Feature contributions (SHAP values):",
        ]

        for sv in top_shap_values:
            direction = "+" if sv.shap_value > 0 else ""
            name = LIMEExplainer.FEATURE_DISPLAY_NAMES.get(
                sv.feature_name,
                sv.feature_name.replace("_", " ").title()
            )
            summary_lines.append(
                f"  - {name}: {direction}{sv.shap_value:.4f}"
            )

        return "\n".join(summary_lines)

    def _calculate_provenance(
        self,
        features: FeatureVector,
        failure_mode: FailureMode,
        shap_values: Dict[str, float],
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        feature_str = "|".join(
            f"{k}:{v:.6f}" for k, v in sorted(features.items())
        )
        shap_str = "|".join(
            f"{k}:{v:.6f}" for k, v in sorted(shap_values.items())
        )
        provenance_str = (
            f"SHAP|{failure_mode.value}|{feature_str}|{shap_str}|"
            f"{self.expected_value:.6f}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def get_global_importance(self) -> Dict[str, float]:
        """
        Calculate global feature importance across background samples.

        Returns:
            Dictionary of feature name to average absolute SHAP value
        """
        global_importance = {name: 0.0 for name in self.feature_names}

        for sample in self.background_data[:20]:  # Limit for efficiency
            shap_values = self._calculate_shap_values(sample)
            for name, value in shap_values.items():
                global_importance[name] += abs(value)

        # Normalize
        num_samples = min(20, len(self.background_data))
        if num_samples > 0:
            for name in global_importance:
                global_importance[name] /= num_samples

        return global_importance


# =============================================================================
# COUNTERFACTUAL EXPLAINER
# =============================================================================

class CounterfactualExplainer:
    """
    Counterfactual explanation generator for PdM predictions.

    Generates "what-if" scenarios showing what changes would result
    in a different prediction outcome.

    Example:
        >>> explainer = CounterfactualExplainer(engine.predict_probability)
        >>> cf = explainer.generate_counterfactual(
        ...     features,
        ...     FailureMode.BEARING_WEAR,
        ...     target_probability=0.1
        ... )
        >>> print(cf.action_recommendations)
    """

    def __init__(
        self,
        prediction_fn: PredictionFunction,
        feature_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """
        Initialize counterfactual explainer.

        Args:
            prediction_fn: Function that takes features and returns probability
            feature_constraints: Min/max bounds for each feature
        """
        self.prediction_fn = prediction_fn
        self.feature_constraints = feature_constraints or {}

        logger.info("CounterfactualExplainer initialized")

    def generate_counterfactual(
        self,
        features: FeatureVector,
        failure_mode: FailureMode,
        target_probability: float = 0.1,
        max_changes: int = 3,
    ) -> CounterfactualExplanation:
        """
        Generate counterfactual explanation.

        Args:
            features: Original feature vector
            failure_mode: Failure mode to explain
            target_probability: Target probability to achieve
            max_changes: Maximum number of features to change

        Returns:
            CounterfactualExplanation with recommended changes
        """
        logger.info(
            f"Generating counterfactual for {failure_mode.value}: "
            f"target={target_probability:.1%}"
        )

        original_prob = self.prediction_fn(features)

        # Find features to change
        feature_changes = {}
        current_features = dict(features)
        current_prob = original_prob

        # Greedy search for feature changes
        for _ in range(max_changes):
            if abs(current_prob - target_probability) < 0.05:
                break

            best_change = None
            best_new_prob = current_prob

            for name, value in current_features.items():
                # Try reducing the feature value
                for delta in [-0.2, -0.1, 0.1, 0.2]:
                    new_value = max(0, min(1, value + delta))

                    if abs(new_value - value) < 0.01:
                        continue

                    test_features = dict(current_features)
                    test_features[name] = new_value
                    new_prob = self.prediction_fn(test_features)

                    # Check if this moves toward target
                    if abs(new_prob - target_probability) < abs(best_new_prob - target_probability):
                        best_change = (name, value, new_value)
                        best_new_prob = new_prob

            if best_change is None:
                break

            name, old_value, new_value = best_change
            feature_changes[name] = (old_value, new_value)
            current_features[name] = new_value
            current_prob = best_new_prob

        # Calculate distance
        distance = sum(
            abs(new - old)
            for old, new in feature_changes.values()
        )

        # Calculate feasibility (closer to original is more feasible)
        feasibility = 1.0 - min(1.0, distance / len(feature_changes)) if feature_changes else 1.0

        # Generate action recommendations
        recommendations = self._generate_recommendations(
            feature_changes,
            failure_mode,
        )

        prediction_id = f"CF-{failure_mode.value}-{datetime.now(timezone.utc).isoformat()}"

        return CounterfactualExplanation(
            prediction_id=prediction_id,
            failure_mode=failure_mode,
            original_probability=original_prob,
            target_probability=current_prob,
            feature_changes=feature_changes,
            distance=distance,
            feasibility_score=feasibility,
            action_recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        feature_changes: Dict[str, Tuple[float, float]],
        failure_mode: FailureMode,
    ) -> List[str]:
        """Generate actionable recommendations from counterfactual."""
        recommendations = []

        action_mapping = {
            "velocity_rms_normalized": "Reduce vibration through balancing/alignment",
            "bearing_defect_indicator": "Replace or lubricate bearings",
            "temperature_normalized": "Improve cooling or reduce load",
            "tan_normalized": "Change oil to reduce acidity",
            "iron_ppm_normalized": "Change oil and investigate wear source",
            "current_unbalance_pct": "Check electrical connections and windings",
            "load_factor": "Reduce operating load",
        }

        for feature, (old, new) in feature_changes.items():
            if new < old:  # Reduction needed
                action = action_mapping.get(
                    feature,
                    f"Reduce {feature.replace('_', ' ')}"
                )
                recommendations.append(action)

        if not recommendations:
            recommendations.append("No specific actions identified")

        return recommendations


# =============================================================================
# INTEGRATED EXPLANATION GENERATOR
# =============================================================================

class MaintenanceExplainer:
    """
    Integrated explainer combining LIME, SHAP, and counterfactuals.

    Provides comprehensive explanations for predictive maintenance decisions.

    Example:
        >>> explainer = MaintenanceExplainer(prediction_engine)
        >>> full_explanation = explainer.explain_prediction(
        ...     features,
        ...     prediction,
        ...     include_counterfactual=True
        ... )
    """

    def __init__(
        self,
        prediction_fn: PredictionFunction,
        background_data: Optional[List[FeatureVector]] = None,
    ) -> None:
        """
        Initialize integrated explainer.

        Args:
            prediction_fn: Prediction function
            background_data: Background samples for SHAP
        """
        self.lime_explainer = LIMEExplainer(prediction_fn)
        self.shap_analyzer = SHAPAnalyzer(prediction_fn, background_data)
        self.counterfactual_explainer = CounterfactualExplainer(prediction_fn)

        logger.info("MaintenanceExplainer initialized with LIME, SHAP, and CF")

    def explain_prediction(
        self,
        features: FeatureVector,
        prediction: FailurePrediction,
        include_lime: bool = True,
        include_shap: bool = True,
        include_counterfactual: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction.

        Args:
            features: Feature vector
            prediction: Failure prediction to explain
            include_lime: Include LIME explanation
            include_shap: Include SHAP explanation
            include_counterfactual: Include counterfactual

        Returns:
            Dictionary containing all requested explanations
        """
        result = {
            "prediction": {
                "failure_mode": prediction.failure_mode.value,
                "probability": prediction.probability,
                "confidence": prediction.confidence,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if include_lime:
            lime_exp = self.lime_explainer.explain(
                features,
                prediction.failure_mode,
                prediction,
            )
            result["lime"] = {
                "explanation_quality": lime_exp.r_squared,
                "top_features": [
                    {
                        "name": c.human_readable_name,
                        "contribution": c.contribution,
                        "direction": c.direction,
                    }
                    for c in lime_exp.feature_contributions[:5]
                ],
                "summary": lime_exp.summary,
                "provenance_hash": lime_exp.provenance_hash,
            }

        if include_shap:
            shap_exp = self.shap_analyzer.explain(
                features,
                prediction.failure_mode,
                prediction,
            )
            result["shap"] = {
                "base_value": shap_exp.base_value,
                "shap_values": [
                    {
                        "feature": sv.feature_name,
                        "value": sv.shap_value,
                    }
                    for sv in shap_exp.shap_values[:5]
                ],
                "summary": shap_exp.summary,
                "provenance_hash": shap_exp.provenance_hash,
            }

        if include_counterfactual:
            cf_exp = self.counterfactual_explainer.generate_counterfactual(
                features,
                prediction.failure_mode,
            )
            result["counterfactual"] = {
                "original_probability": cf_exp.original_probability,
                "target_probability": cf_exp.target_probability,
                "required_changes": {
                    k: {"from": v[0], "to": v[1]}
                    for k, v in cf_exp.feature_changes.items()
                },
                "feasibility": cf_exp.feasibility_score,
                "recommendations": cf_exp.action_recommendations,
            }

        return result

    def get_global_explanation(
        self,
        failure_modes: Optional[List[FailureMode]] = None,
    ) -> GlobalExplanation:
        """
        Generate global explanation of the model.

        Args:
            failure_modes: Failure modes to include (all if None)

        Returns:
            GlobalExplanation with global feature importance
        """
        # Get global SHAP importance
        global_importance = self.shap_analyzer.get_global_importance()

        # Placeholder for per-mode importance
        importance_by_mode = {}

        # Determine feature monotonicity (simplified)
        monotonicity = {
            name: "increasing" if name in [
                "velocity_rms_normalized",
                "bearing_defect_indicator",
                "tan_normalized",
                "temperature_normalized",
            ] else "unknown"
            for name in global_importance.keys()
        }

        # Identify top interactions
        interactions = []  # Would come from SHAP interaction analysis

        summary = (
            "Global model explanation:\n"
            f"Most important features: "
            f"{', '.join(sorted(global_importance.keys(), key=lambda x: global_importance[x], reverse=True)[:5])}"
        )

        return GlobalExplanation(
            model_id="gl013_failure_pred",
            feature_importance=global_importance,
            feature_importance_by_mode=importance_by_mode,
            feature_interactions=interactions,
            monotonicity=monotonicity,
            summary=summary,
        )
