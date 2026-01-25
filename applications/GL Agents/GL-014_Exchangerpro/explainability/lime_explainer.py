# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - LIME Explainer for Fouling Analysis

Local Interpretable Model-agnostic Explanations for heat exchanger
fouling predictions. Provides local surrogate model explanations
and counterfactual analysis.

Zero-Hallucination Principle:
- All LIME weights are computed from deterministic mathematical formulas
- No LLM is used for numeric calculations
- Provenance tracking via SHA-256 hashes

Reference: Ribeiro et al., "Why Should I Trust You?:
Explaining the Predictions of Any Classifier", KDD 2016.

Author: GreenLang AI Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging
import time
import uuid

import numpy as np

try:
    from lime import lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

from .explanation_schemas import (
    ConfidenceBounds,
    ConfidenceLevel,
    CounterfactualExplanation,
    ExplanationType,
    FeatureCategory,
    FeatureContribution,
    LocalExplanation,
    PredictionType,
    ExplanationStabilityMetrics,
)
from .shap_explainer import FOULING_FEATURE_CATEGORIES

logger = logging.getLogger(__name__)


@dataclass
class LIMEConfig:
    """Configuration for LIME explainer."""
    num_features: int = 10
    num_samples: int = 500
    kernel_width: float = 0.75
    discretize_continuous: bool = True
    feature_selection: str = "auto"  # "auto", "forward_selection", "lasso"
    random_seed: int = 42
    stability_neighbors: int = 10
    stability_perturbation: float = 0.01
    counterfactual_steps: int = 10
    cache_enabled: bool = True


@dataclass
class LIMEResult:
    """Result from LIME analysis."""
    explanation_id: str
    feature_names: List[str]
    feature_weights: Dict[str, float]
    feature_values: np.ndarray
    local_prediction: float
    intercept: float
    r2_score: float
    num_features_used: int
    explanation_text: List[str]
    stability_score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    computation_hash: str = ""
    computation_time_ms: float = 0.0


class FoulingLIMEExplainer:
    """
    LIME-based explainer for heat exchanger fouling predictions.

    Provides local interpretable explanations by fitting a linear
    surrogate model in the neighborhood of a specific prediction.
    Also supports counterfactual analysis for "what-if" scenarios.

    Features:
    - Local interpretable explanations
    - Surrogate model explanations with R2 scores
    - Counterfactual analysis ("what needs to change?")
    - Stability assessment across similar operating points
    - Engineering-focused interpretations

    Example:
        >>> config = LIMEConfig(num_features=10)
        >>> explainer = FoulingLIMEExplainer(config)
        >>> result = explainer.explain_prediction(
        ...     model=fouling_model,
        ...     features=input_features,
        ...     feature_names=feature_names,
        ...     prediction_type=PredictionType.FOULING_FACTOR
        ... )
        >>> for line in result.explanation_text:
        ...     print(line)
    """

    VERSION = "1.0.0"
    METHODOLOGY_REFERENCE = "Ribeiro et al., KDD 2016"

    def __init__(self, config: Optional[LIMEConfig] = None) -> None:
        """
        Initialize LIME explainer.

        Args:
            config: Configuration options for LIME computation
        """
        self.config = config or LIMEConfig()
        self._explainer = None
        self._training_data: Optional[np.ndarray] = None
        self._feature_names: List[str] = []
        self._cache: Dict[str, LIMEResult] = {}
        np.random.seed(self.config.random_seed)

        logger.info(
            f"FoulingLIMEExplainer initialized with config: "
            f"num_features={self.config.num_features}, "
            f"num_samples={self.config.num_samples}"
        )

    def explain_prediction(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        prediction_type: PredictionType,
        exchanger_id: str = "unknown",
        training_data: Optional[np.ndarray] = None,
    ) -> LocalExplanation:
        """
        Generate local LIME explanation for a single prediction.

        Args:
            model: Trained model with predict method
            features: Feature values for the instance (1D array)
            feature_names: Names of features
            prediction_type: Type of prediction being explained
            exchanger_id: Identifier of the heat exchanger
            training_data: Optional training data for LIME

        Returns:
            LocalExplanation with feature contributions and confidence

        Raises:
            ValueError: If features and feature_names have mismatched lengths
        """
        start_time = time.time()

        if len(features) != len(feature_names):
            raise ValueError(
                f"Features length ({len(features)}) != feature_names length ({len(feature_names)})"
            )

        self._feature_names = feature_names

        # Ensure features is 1D for LIME
        if features.ndim > 1:
            features = features.flatten()

        # Set up training data
        if training_data is not None:
            self._training_data = training_data
        elif self._training_data is None:
            self._training_data = self._generate_training_data(features)

        # Create prediction function wrapper
        def predict_fn(X: np.ndarray) -> np.ndarray:
            return model.predict(X)

        # Compute LIME explanation
        lime_result = self._compute_lime_explanation(
            features, feature_names, predict_fn, prediction_type
        )

        # Get prediction
        prediction_value = float(model.predict(features.reshape(1, -1))[0])

        # Build feature contributions
        contributions = self._build_feature_contributions(
            lime_result, features, feature_names
        )

        # Assess stability
        stability_metrics = self._assess_stability(
            model, features, feature_names, lime_result
        )

        # Determine confidence based on R2 and stability
        confidence = self._determine_confidence(
            lime_result.r2_score, stability_metrics.stability_score
        )

        # Identify top drivers
        sorted_contributions = sorted(
            contributions, key=lambda x: abs(x.contribution), reverse=True
        )
        top_positive = [
            c.feature_name for c in sorted_contributions
            if c.contribution > 0
        ][:5]
        top_negative = [
            c.feature_name for c in sorted_contributions
            if c.contribution < 0
        ][:5]

        computation_time = (time.time() - start_time) * 1000
        explanation_id = str(uuid.uuid4())

        local_explanation = LocalExplanation(
            explanation_id=explanation_id,
            exchanger_id=exchanger_id,
            timestamp=datetime.now(timezone.utc),
            prediction_type=prediction_type,
            prediction_value=prediction_value,
            base_value=lime_result.intercept,
            feature_contributions=contributions,
            top_positive_drivers=top_positive,
            top_negative_drivers=top_negative,
            explanation_method=ExplanationType.LIME,
            local_accuracy=lime_result.r2_score,
            stability_score=stability_metrics.stability_score,
            confidence=confidence,
            computation_time_ms=computation_time,
            provenance_hash=lime_result.computation_hash,
            methodology_version=self.VERSION,
        )

        logger.info(
            f"LIME explanation generated for exchanger {exchanger_id} "
            f"in {computation_time:.2f}ms (R2={lime_result.r2_score:.4f})"
        )

        return local_explanation

    def generate_counterfactual(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        target_prediction: float,
        exchanger_id: str = "unknown",
        changeable_features: Optional[List[str]] = None,
    ) -> CounterfactualExplanation:
        """
        Generate counterfactual explanation: what needs to change?

        Finds the minimal feature changes needed to achieve a target
        prediction value (e.g., lower fouling factor).

        Args:
            model: Trained model with predict method
            features: Current feature values
            feature_names: Names of features
            target_prediction: Desired prediction value
            exchanger_id: Heat exchanger identifier
            changeable_features: Features that can be modified (optional)

        Returns:
            CounterfactualExplanation with required changes
        """
        start_time = time.time()

        if features.ndim > 1:
            features = features.flatten()

        original_prediction = float(model.predict(features.reshape(1, -1))[0])

        # Determine which features can be changed
        if changeable_features is None:
            # Default: operating conditions and hydraulic features can be changed
            changeable_features = [
                name for name in feature_names
                if self._get_feature_category(name) in [
                    FeatureCategory.OPERATING_CONDITIONS,
                    FeatureCategory.HYDRAULIC,
                ]
            ]

        # Find counterfactual using gradient-free optimization
        feature_changes, counterfactual_features = self._find_counterfactual(
            model, features, feature_names, target_prediction,
            changeable_features
        )

        # Compute counterfactual prediction
        cf_prediction = float(model.predict(counterfactual_features.reshape(1, -1))[0])

        # Assess feasibility
        feasibility_score = self._assess_feasibility(feature_changes, feature_names)

        # Generate explanation text
        explanation_text = self._generate_counterfactual_text(
            feature_changes, original_prediction, cf_prediction, target_prediction
        )

        # Compute provenance hash
        provenance_data = {
            "original_features": features.tolist(),
            "target_prediction": target_prediction,
            "feature_changes": {k: list(v) for k, v in feature_changes.items()},
            "version": self.VERSION,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        computation_time = (time.time() - start_time) * 1000

        # Determine confidence
        achieved_target = abs(cf_prediction - target_prediction) < abs(original_prediction - target_prediction) * 0.5
        confidence = ConfidenceLevel.HIGH if achieved_target and feasibility_score > 0.7 else \
                     ConfidenceLevel.MEDIUM if achieved_target else ConfidenceLevel.LOW

        return CounterfactualExplanation(
            explanation_id=str(uuid.uuid4()),
            exchanger_id=exchanger_id,
            original_prediction=original_prediction,
            target_prediction=cf_prediction,
            feature_changes=feature_changes,
            feasibility_score=feasibility_score,
            cost_estimate=None,  # Would need cost model
            explanation_text=explanation_text,
            confidence=confidence,
            provenance_hash=provenance_hash,
            timestamp=datetime.now(timezone.utc),
        )

    def explain_surrogate_model(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Get detailed information about the local surrogate model.

        Args:
            model: Trained model
            features: Feature values for the instance
            feature_names: Names of features

        Returns:
            Dictionary with surrogate model details
        """
        if features.ndim > 1:
            features = features.flatten()

        if self._training_data is None:
            self._training_data = self._generate_training_data(features)

        def predict_fn(X: np.ndarray) -> np.ndarray:
            return model.predict(X)

        # Compute LIME explanation
        lime_result = self._compute_lime_explanation(
            features, feature_names, predict_fn, PredictionType.FOULING_FACTOR
        )

        # Build surrogate model equation
        equation_terms = []
        for name, weight in sorted(lime_result.feature_weights.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(weight) > 1e-6:
                sign = "+" if weight > 0 else "-"
                equation_terms.append(f"{sign} {abs(weight):.4f} * {name}")

        equation = f"f(x) = {lime_result.intercept:.4f} " + " ".join(equation_terms)

        return {
            "equation": equation,
            "intercept": lime_result.intercept,
            "coefficients": lime_result.feature_weights,
            "r2_score": lime_result.r2_score,
            "num_features": lime_result.num_features_used,
            "kernel_width": self.config.kernel_width,
            "num_samples": self.config.num_samples,
            "methodology": self.METHODOLOGY_REFERENCE,
        }

    def _compute_lime_explanation(
        self,
        features: np.ndarray,
        feature_names: List[str],
        predict_fn: Callable,
        prediction_type: PredictionType,
    ) -> LIMEResult:
        """Compute LIME explanation."""
        start_time = time.time()

        if HAS_LIME:
            result = self._compute_with_lime_library(
                features, feature_names, predict_fn
            )
        else:
            result = self._compute_weighted_regression(
                features, feature_names, predict_fn
            )

        result.computation_time_ms = (time.time() - start_time) * 1000
        return result

    def _compute_with_lime_library(
        self,
        features: np.ndarray,
        feature_names: List[str],
        predict_fn: Callable,
    ) -> LIMEResult:
        """Compute LIME using the lime library."""
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=self._training_data,
            feature_names=feature_names,
            mode="regression",
            discretize_continuous=self.config.discretize_continuous,
            kernel_width=self.config.kernel_width * np.sqrt(len(features)),
            feature_selection=self.config.feature_selection,
            random_state=self.config.random_seed,
        )

        explanation = explainer.explain_instance(
            features,
            predict_fn,
            num_features=self.config.num_features,
            num_samples=self.config.num_samples,
        )

        # Extract weights and build result
        feature_weights = {}
        explanation_text = []

        for feat, weight in explanation.as_list():
            feature_weights[feat] = round(weight, 6)
            direction = "increases" if weight > 0 else "decreases"
            explanation_text.append(
                f"{feat}: {direction} fouling prediction by {abs(weight):.4f}"
            )

        local_prediction = float(predict_fn(features.reshape(1, -1))[0])
        intercept = float(explanation.intercept[0]) if hasattr(explanation, 'intercept') else 0.0
        r2_score = float(explanation.score) if hasattr(explanation, 'score') else 0.0

        # Compute provenance hash
        provenance_data = {
            "features": features.tolist(),
            "weights": feature_weights,
            "prediction": local_prediction,
            "version": self.VERSION,
        }
        computation_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return LIMEResult(
            explanation_id=str(uuid.uuid4()),
            feature_names=feature_names,
            feature_weights=feature_weights,
            feature_values=features,
            local_prediction=round(local_prediction, 6),
            intercept=round(intercept, 6),
            r2_score=round(r2_score, 4),
            num_features_used=len(feature_weights),
            explanation_text=explanation_text,
            stability_score=0.0,  # Will be computed later
            computation_hash=computation_hash,
        )

    def _compute_weighted_regression(
        self,
        features: np.ndarray,
        feature_names: List[str],
        predict_fn: Callable,
    ) -> LIMEResult:
        """Fallback: compute weighted linear regression for explanation."""
        n_samples = self.config.num_samples

        # Generate perturbed samples
        perturbations = np.random.randn(n_samples, len(features)) * 0.1
        X_perturbed = features + perturbations * np.abs(features + 1e-10)
        X_perturbed = np.maximum(X_perturbed, 0.01)

        # Get predictions
        y = predict_fn(X_perturbed)

        # Compute weights (exponential kernel)
        distances = np.sqrt(np.sum(perturbations ** 2, axis=1))
        kernel_width = self.config.kernel_width * np.sqrt(len(features))
        weights = np.exp(-distances ** 2 / (2 * kernel_width ** 2))

        # Weighted linear regression
        try:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            model.fit(X_perturbed, y, sample_weight=weights)
            coefficients = model.coef_
            intercept = model.intercept_
            r2_score = model.score(X_perturbed, y, sample_weight=weights)
        except ImportError:
            # Manual weighted least squares
            coefficients, intercept, r2_score = self._manual_weighted_regression(
                X_perturbed, y, weights
            )

        # Build feature weights
        feature_weights = {}
        explanation_text = []

        indices = np.argsort(np.abs(coefficients))[::-1][:self.config.num_features]
        for i in indices:
            weight = coefficients[i]
            name = feature_names[i]
            feature_weights[name] = round(float(weight), 6)
            direction = "increases" if weight > 0 else "decreases"
            explanation_text.append(
                f"{name}: {direction} fouling prediction by {abs(weight):.4f}"
            )

        local_prediction = float(predict_fn(features.reshape(1, -1))[0])

        # Compute provenance hash
        provenance_data = {
            "features": features.tolist(),
            "weights": feature_weights,
            "prediction": local_prediction,
            "version": self.VERSION,
        }
        computation_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return LIMEResult(
            explanation_id=str(uuid.uuid4()),
            feature_names=feature_names,
            feature_weights=feature_weights,
            feature_values=features,
            local_prediction=round(local_prediction, 6),
            intercept=round(float(intercept), 6),
            r2_score=round(float(r2_score), 4),
            num_features_used=len(feature_weights),
            explanation_text=explanation_text,
            stability_score=0.0,
            computation_hash=computation_hash,
        )

    def _manual_weighted_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """Manual weighted least squares implementation."""
        n_samples = len(y)
        W = np.diag(weights)
        X_aug = np.column_stack([np.ones(n_samples), X])

        try:
            beta = np.linalg.solve(
                X_aug.T @ W @ X_aug + 0.01 * np.eye(X_aug.shape[1]),
                X_aug.T @ W @ y
            )
            intercept = beta[0]
            coefficients = beta[1:]

            # Compute R2
            y_pred = X_aug @ beta
            ss_res = np.sum(weights * (y - y_pred) ** 2)
            ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
            r2_score = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        except np.linalg.LinAlgError:
            coefficients = np.zeros(X.shape[1])
            intercept = 0.0
            r2_score = 0.0

        return coefficients, intercept, r2_score

    def _generate_training_data(self, reference: np.ndarray) -> np.ndarray:
        """Generate synthetic training data around reference point."""
        n_samples = self.config.num_samples
        n_features = len(reference)

        # Generate variations
        variations = np.random.randn(n_samples, n_features) * 0.2
        training_data = reference + variations * (np.abs(reference) + 1e-10)

        # Ensure positive values where appropriate
        training_data = np.maximum(training_data, 0.01)

        return training_data

    def _build_feature_contributions(
        self,
        lime_result: LIMEResult,
        features: np.ndarray,
        feature_names: List[str],
    ) -> List[FeatureContribution]:
        """Build list of feature contributions from LIME weights."""
        contributions = []
        total_abs = sum(abs(w) for w in lime_result.feature_weights.values()) or 1.0

        for i, name in enumerate(feature_names):
            # LIME may return discretized feature names, so try to match
            weight = lime_result.feature_weights.get(name, 0.0)

            # Also check for discretized versions
            if weight == 0.0:
                for lime_name, lime_weight in lime_result.feature_weights.items():
                    if name in lime_name:
                        weight = lime_weight
                        break

            direction = "positive" if weight > 0 else "negative" if weight < 0 else "neutral"

            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=float(features[i]),
                baseline_value=None,
                contribution=round(float(weight), 6),
                contribution_percentage=round(float(weight / total_abs * 100), 4),
                direction=direction,
                category=self._get_feature_category(name),
                unit=self._get_feature_unit(name),
                is_anomalous=False,
            ))

        return contributions

    def _assess_stability(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        original_result: LIMEResult,
    ) -> ExplanationStabilityMetrics:
        """Assess stability of LIME explanations across similar inputs."""
        n_neighbors = self.config.stability_neighbors
        perturbation = self.config.stability_perturbation

        # Generate neighboring points
        neighbors = features + np.random.randn(n_neighbors, len(features)) * perturbation * (np.abs(features) + 1e-10)
        neighbors = np.maximum(neighbors, 0.01)

        def predict_fn(X: np.ndarray) -> np.ndarray:
            return model.predict(X)

        # Compute LIME for neighbors
        neighbor_weights = []
        for neighbor in neighbors:
            try:
                if HAS_LIME:
                    result = self._compute_with_lime_library(neighbor, feature_names, predict_fn)
                else:
                    result = self._compute_weighted_regression(neighbor, feature_names, predict_fn)
                neighbor_weights.append(result.feature_weights)
            except:
                continue

        if len(neighbor_weights) < 2:
            return ExplanationStabilityMetrics(
                stability_score=0.5,
                feature_ranking_stability=0.5,
                contribution_variance=0.5,
                neighboring_points_analyzed=len(neighbor_weights),
                stability_method="neighborhood_sampling",
            )

        # Compute ranking stability
        original_ranking = sorted(
            original_result.feature_weights.keys(),
            key=lambda k: abs(original_result.feature_weights.get(k, 0)),
            reverse=True
        )[:5]

        ranking_agreements = []
        for nw in neighbor_weights:
            neighbor_ranking = sorted(
                nw.keys(),
                key=lambda k: abs(nw.get(k, 0)),
                reverse=True
            )[:5]
            agreement = len(set(original_ranking) & set(neighbor_ranking)) / 5
            ranking_agreements.append(agreement)

        feature_ranking_stability = float(np.mean(ranking_agreements))

        # Compute contribution variance
        all_weights = []
        for nw in neighbor_weights:
            weights = [nw.get(name, 0.0) for name in feature_names[:self.config.num_features]]
            all_weights.append(weights)

        if all_weights:
            contribution_variance = float(np.mean(np.std(all_weights, axis=0)))
        else:
            contribution_variance = 0.5

        # Overall stability
        stability_score = float(np.clip(
            0.6 * feature_ranking_stability + 0.4 * (1 - min(contribution_variance, 1.0)),
            0.0, 1.0
        ))

        return ExplanationStabilityMetrics(
            stability_score=round(stability_score, 4),
            feature_ranking_stability=round(feature_ranking_stability, 4),
            contribution_variance=round(contribution_variance, 6),
            neighboring_points_analyzed=len(neighbor_weights),
            stability_method="neighborhood_sampling",
        )

    def _determine_confidence(
        self,
        r2_score: float,
        stability_score: float,
    ) -> ConfidenceLevel:
        """Determine confidence level based on R2 and stability."""
        quality_score = 0.5 * r2_score + 0.5 * stability_score

        if quality_score >= 0.85:
            return ConfidenceLevel.HIGH
        elif quality_score >= 0.65:
            return ConfidenceLevel.MEDIUM
        elif quality_score >= 0.40:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _find_counterfactual(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        target_prediction: float,
        changeable_features: List[str],
    ) -> Tuple[Dict[str, Tuple[float, float]], np.ndarray]:
        """Find counterfactual through gradient-free optimization."""
        changeable_indices = [
            i for i, name in enumerate(feature_names) if name in changeable_features
        ]

        if not changeable_indices:
            # No features can be changed
            return {}, features.copy()

        best_features = features.copy()
        best_distance = float('inf')
        original_prediction = float(model.predict(features.reshape(1, -1))[0])

        # Simple grid search over changeable features
        for _ in range(self.config.counterfactual_steps):
            candidate = features.copy()

            # Perturb changeable features
            for idx in changeable_indices:
                direction = 1 if target_prediction > original_prediction else -1
                perturbation = direction * np.random.uniform(0.01, 0.1) * (np.abs(features[idx]) + 1e-10)
                candidate[idx] = max(0.01, features[idx] + perturbation)

            # Evaluate
            pred = float(model.predict(candidate.reshape(1, -1))[0])
            distance = abs(pred - target_prediction)

            if distance < best_distance:
                best_distance = distance
                best_features = candidate.copy()

        # Build feature changes dictionary
        feature_changes = {}
        for idx in changeable_indices:
            if abs(best_features[idx] - features[idx]) > 1e-6:
                feature_changes[feature_names[idx]] = (
                    float(features[idx]),
                    float(best_features[idx])
                )

        return feature_changes, best_features

    def _assess_feasibility(
        self,
        feature_changes: Dict[str, Tuple[float, float]],
        feature_names: List[str],
    ) -> float:
        """Assess feasibility of counterfactual changes."""
        if not feature_changes:
            return 0.0

        feasibility_scores = []
        for name, (original, new) in feature_changes.items():
            # Calculate relative change
            relative_change = abs(new - original) / (abs(original) + 1e-10)

            # Smaller changes are more feasible
            if relative_change < 0.1:
                score = 1.0
            elif relative_change < 0.25:
                score = 0.8
            elif relative_change < 0.5:
                score = 0.5
            else:
                score = 0.2

            # Operating conditions are easier to change
            category = self._get_feature_category(name)
            if category == FeatureCategory.OPERATING_CONDITIONS:
                score *= 1.2
            elif category == FeatureCategory.GEOMETRY:
                score *= 0.3  # Geometry changes are expensive

            feasibility_scores.append(min(score, 1.0))

        return float(np.mean(feasibility_scores))

    def _generate_counterfactual_text(
        self,
        feature_changes: Dict[str, Tuple[float, float]],
        original_prediction: float,
        counterfactual_prediction: float,
        target_prediction: float,
    ) -> str:
        """Generate human-readable counterfactual explanation."""
        if not feature_changes:
            return "No feasible changes found to achieve the target prediction."

        change_descriptions = []
        for name, (original, new) in feature_changes.items():
            direction = "increase" if new > original else "decrease"
            change_pct = abs(new - original) / (abs(original) + 1e-10) * 100
            change_descriptions.append(
                f"{direction} {name} from {original:.4f} to {new:.4f} ({change_pct:.1f}% change)"
            )

        changes_text = "; ".join(change_descriptions)
        prediction_change = counterfactual_prediction - original_prediction

        return (
            f"To move the prediction from {original_prediction:.4f} toward {target_prediction:.4f}: "
            f"{changes_text}. "
            f"Expected new prediction: {counterfactual_prediction:.4f} "
            f"(change of {prediction_change:+.4f})."
        )

    def _get_feature_category(self, name: str) -> FeatureCategory:
        """Get category for a feature."""
        if name in FOULING_FEATURE_CATEGORIES:
            return FOULING_FEATURE_CATEGORIES[name]

        name_lower = name.lower()
        if any(t in name_lower for t in ['temp', 't_hot', 't_cold', 'lmtd']):
            return FeatureCategory.THERMAL
        elif any(h in name_lower for h in ['pressure', 'delta_p', 'flow', 'velocity']):
            return FeatureCategory.HYDRAULIC
        elif any(f in name_lower for f in ['viscosity', 'density', 'cp']):
            return FeatureCategory.FLUID_PROPERTIES
        elif any(o in name_lower for o in ['hour', 'cycle', 'day', 'load']):
            return FeatureCategory.OPERATING_CONDITIONS
        elif any(g in name_lower for g in ['diameter', 'length', 'area']):
            return FeatureCategory.GEOMETRY
        else:
            return FeatureCategory.OPERATING_CONDITIONS

    def _get_feature_unit(self, name: str) -> Optional[str]:
        """Get unit of measurement for a feature."""
        units = {
            "delta_T": "K",
            "LMTD": "K",
            "delta_P": "kPa",
            "delta_P_normalized": "-",
            "flow_rate_hot": "kg/s",
            "flow_rate_cold": "kg/s",
            "velocity_hot": "m/s",
            "velocity_cold": "m/s",
            "fouling_factor": "m2K/W",
            "heat_duty": "kW",
            "operating_hours": "h",
            "days_since_cleaning": "days",
        }
        return units.get(name)

    def clear_cache(self) -> None:
        """Clear the explanation cache."""
        self._cache.clear()
        self._training_data = None
        logger.info("LIME explanation cache cleared")


# Convenience functions
def explain_fouling_with_lime(
    model: Any,
    features: np.ndarray,
    feature_names: List[str],
    prediction_type: PredictionType = PredictionType.FOULING_FACTOR,
    exchanger_id: str = "unknown",
    config: Optional[LIMEConfig] = None,
) -> LocalExplanation:
    """
    Convenience function to generate local LIME explanation.

    Args:
        model: Trained fouling prediction model
        features: Feature values for the instance
        feature_names: Names of features
        prediction_type: Type of prediction
        exchanger_id: Heat exchanger identifier
        config: Optional LIME configuration

    Returns:
        LocalExplanation with feature contributions
    """
    explainer = FoulingLIMEExplainer(config)
    return explainer.explain_prediction(
        model, features, feature_names, prediction_type, exchanger_id
    )


def generate_counterfactual_explanation(
    model: Any,
    features: np.ndarray,
    feature_names: List[str],
    target_prediction: float,
    exchanger_id: str = "unknown",
    changeable_features: Optional[List[str]] = None,
    config: Optional[LIMEConfig] = None,
) -> CounterfactualExplanation:
    """
    Convenience function to generate counterfactual explanation.

    Args:
        model: Trained fouling prediction model
        features: Current feature values
        feature_names: Names of features
        target_prediction: Desired prediction value
        exchanger_id: Heat exchanger identifier
        changeable_features: Features that can be modified
        config: Optional LIME configuration

    Returns:
        CounterfactualExplanation with required changes
    """
    explainer = FoulingLIMEExplainer(config)
    return explainer.generate_counterfactual(
        model, features, feature_names, target_prediction,
        exchanger_id, changeable_features
    )


def validate_lime_explanation(
    result: LIMEResult,
    min_r2: float = 0.5,
    min_stability: float = 0.6,
) -> Tuple[bool, List[str]]:
    """
    Validate LIME explanation quality.

    Args:
        result: LIME analysis result
        min_r2: Minimum acceptable R2 score
        min_stability: Minimum acceptable stability score

    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = []
    is_valid = True

    if result.r2_score < min_r2:
        warnings.append(f"Low R2 score: {result.r2_score:.4f} < {min_r2}")
        is_valid = False

    if result.stability_score < min_stability:
        warnings.append(f"Low stability score: {result.stability_score:.4f} < {min_stability}")
        is_valid = False

    if result.num_features_used < 3:
        warnings.append(f"Few features used: {result.num_features_used}")

    return is_valid, warnings


def compare_lime_explanations(
    result1: LIMEResult,
    result2: LIMEResult,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Compare two LIME explanations.

    Args:
        result1: First LIME result
        result2: Second LIME result
        top_k: Number of top features to compare

    Returns:
        Comparison statistics
    """
    # Get top features from each
    top1 = sorted(
        result1.feature_weights.keys(),
        key=lambda k: abs(result1.feature_weights.get(k, 0)),
        reverse=True
    )[:top_k]

    top2 = sorted(
        result2.feature_weights.keys(),
        key=lambda k: abs(result2.feature_weights.get(k, 0)),
        reverse=True
    )[:top_k]

    # Compute overlap
    overlap = len(set(top1) & set(top2)) / top_k

    # Compare weights for common features
    common_features = set(result1.feature_weights.keys()) & set(result2.feature_weights.keys())
    weight_correlations = []
    for feat in common_features:
        w1 = result1.feature_weights[feat]
        w2 = result2.feature_weights[feat]
        weight_correlations.append((w1, w2))

    if weight_correlations:
        w1_arr = np.array([w[0] for w in weight_correlations])
        w2_arr = np.array([w[1] for w in weight_correlations])
        corr = np.corrcoef(w1_arr, w2_arr)[0, 1] if len(w1_arr) > 1 else 1.0
    else:
        corr = 0.0

    return {
        "top_k_overlap": overlap,
        "weight_correlation": float(corr) if not np.isnan(corr) else 0.0,
        "r2_difference": abs(result1.r2_score - result2.r2_score),
        "prediction_difference": abs(result1.local_prediction - result2.local_prediction),
    }
