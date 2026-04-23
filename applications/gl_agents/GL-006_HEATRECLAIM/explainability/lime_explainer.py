"""
GL-006 HEATRECLAIM - LIME Explainer

Local Interpretable Model-agnostic Explanations for
heat recovery optimization decisions. Provides local
surrogate explanations around specific design points.

Reference: Ribeiro et al., "Why Should I Trust You?:
Explaining the Predictions of Any Classifier", KDD 2016.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Callable
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

from ..core.schemas import (
    HeatStream,
    HENDesign,
    OptimizationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class LIMEResult:
    """Result from LIME analysis."""

    feature_names: List[str]
    feature_weights: Dict[str, float]
    local_prediction: float
    intercept: float
    score: float  # R² of local surrogate
    num_features_used: int
    explanation_text: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    computation_hash: str = ""


class LIMEExplainer:
    """
    LIME-based explainer for heat recovery optimization.

    Creates local linear approximations around specific
    design points to explain why certain optimization
    decisions were made.

    Use cases:
    - Explain why a specific exchanger match was selected
    - Understand local sensitivity to parameter changes
    - Validate optimization reasonableness

    Example:
        >>> explainer = LIMEExplainer()
        >>> result = explainer.explain_design(design, streams)
        >>> for line in result.explanation_text:
        ...     print(line)
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        num_features: int = 10,
        num_samples: int = 500,
        kernel_width: float = 0.75,
        discretize_continuous: bool = True,
    ) -> None:
        """
        Initialize LIME explainer.

        Args:
            num_features: Number of features in explanation
            num_samples: Number of perturbed samples
            kernel_width: Width of exponential kernel
            discretize_continuous: Discretize continuous features
        """
        self.num_features = num_features
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.discretize_continuous = discretize_continuous
        self._explainer = None
        self._feature_names: List[str] = []
        self._training_data: Optional[np.ndarray] = None

    def explain_design(
        self,
        design: HENDesign,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        prediction_fn: Optional[Callable] = None,
    ) -> LIMEResult:
        """
        Explain HEN design using LIME.

        Args:
            design: The HEN design to explain
            hot_streams: Hot process streams
            cold_streams: Cold process streams
            prediction_fn: Optional custom prediction function

        Returns:
            LIMEResult with local explanation
        """
        # Extract features
        features, feature_names = self._extract_features(
            hot_streams, cold_streams, design
        )
        self._feature_names = feature_names

        # Generate training data if not exists
        if self._training_data is None:
            self._training_data = self._generate_training_data(features)

        # Create prediction function if not provided
        if prediction_fn is None:
            prediction_fn = self._create_default_predictor(design)

        # Calculate LIME explanation
        if HAS_LIME:
            result = self._calculate_lime_explanation(
                features, feature_names, prediction_fn
            )
        else:
            # Fallback: weighted linear regression
            result = self._calculate_fallback_explanation(
                features, feature_names, prediction_fn
            )

        return result

    def explain_match(
        self,
        hot_stream: HeatStream,
        cold_stream: HeatStream,
        match_duty_kW: float,
        delta_t_min: float = 10.0,
    ) -> LIMEResult:
        """
        Explain a specific exchanger match selection.

        Args:
            hot_stream: Hot stream in match
            cold_stream: Cold stream in match
            match_duty_kW: Heat duty of match
            delta_t_min: Minimum approach temperature

        Returns:
            LIMEResult explaining the match
        """
        # Create feature vector for match
        features = np.array([
            hot_stream.T_supply_C,
            hot_stream.T_target_C,
            hot_stream.m_dot_kg_s,
            hot_stream.Cp_kJ_kgK,
            cold_stream.T_supply_C,
            cold_stream.T_target_C,
            cold_stream.m_dot_kg_s,
            cold_stream.Cp_kJ_kgK,
            delta_t_min,
            match_duty_kW,
        ])

        feature_names = [
            "hot_T_supply", "hot_T_target", "hot_flow", "hot_Cp",
            "cold_T_supply", "cold_T_target", "cold_flow", "cold_Cp",
            "delta_t_min", "match_duty",
        ]

        self._feature_names = feature_names

        if self._training_data is None:
            self._training_data = self._generate_training_data(features)

        # Create match prediction function
        def match_predictor(X: np.ndarray) -> np.ndarray:
            # Predict match quality/feasibility
            predictions = []
            for row in X:
                hot_T_in, hot_T_out = row[0], row[1]
                cold_T_in, cold_T_out = row[4], row[5]
                dt_min = row[8]

                # Check temperature driving force
                dt1 = hot_T_in - cold_T_out
                dt2 = hot_T_out - cold_T_in

                if dt1 >= dt_min and dt2 >= dt_min:
                    # Feasible match
                    quality = min(dt1, dt2) / dt_min
                else:
                    # Infeasible
                    quality = 0.0

                predictions.append(quality)

            return np.array(predictions)

        if HAS_LIME:
            return self._calculate_lime_explanation(
                features, feature_names, match_predictor
            )
        else:
            return self._calculate_fallback_explanation(
                features, feature_names, match_predictor
            )

    def _extract_features(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        design: HENDesign,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features for LIME analysis."""
        features = []
        names = []

        # Aggregate hot stream features
        if hot_streams:
            features.extend([
                np.mean([s.T_supply_C for s in hot_streams]),
                np.mean([s.T_target_C for s in hot_streams]),
                sum([s.duty_kW for s in hot_streams]),
                sum([s.FCp_kW_K for s in hot_streams]),
            ])
            names.extend([
                "avg_hot_T_supply", "avg_hot_T_target",
                "total_hot_duty", "total_hot_FCp",
            ])

        # Aggregate cold stream features
        if cold_streams:
            features.extend([
                np.mean([s.T_supply_C for s in cold_streams]),
                np.mean([s.T_target_C for s in cold_streams]),
                sum([s.duty_kW for s in cold_streams]),
                sum([s.FCp_kW_K for s in cold_streams]),
            ])
            names.extend([
                "avg_cold_T_supply", "avg_cold_T_target",
                "total_cold_duty", "total_cold_FCp",
            ])

        # Design outputs
        features.extend([
            design.total_heat_recovered_kW,
            design.exchanger_count,
            design.hot_utility_required_kW,
            design.cold_utility_required_kW,
        ])
        names.extend([
            "heat_recovered", "num_exchangers",
            "hot_utility", "cold_utility",
        ])

        return np.array(features), names

    def _generate_training_data(self, reference: np.ndarray) -> np.ndarray:
        """Generate synthetic training data around reference point."""
        n_samples = self.num_samples
        n_features = len(reference)

        # Generate variations
        variations = np.random.randn(n_samples, n_features) * 0.2
        training_data = reference + variations * np.abs(reference)

        # Ensure positive values where appropriate
        training_data = np.maximum(training_data, 0.01)

        return training_data

    def _create_default_predictor(
        self,
        design: HENDesign,
    ) -> Callable:
        """Create default prediction function based on design."""
        reference_heat_recovered = design.total_heat_recovered_kW

        def predictor(X: np.ndarray) -> np.ndarray:
            # Simple linear model: heat recovery based on features
            predictions = []
            for row in X:
                # Estimate heat recovery potential
                if len(row) >= 8:
                    hot_duty = row[2] if len(row) > 2 else 0
                    cold_duty = row[6] if len(row) > 6 else 0
                    potential = min(hot_duty, cold_duty) * 0.7
                else:
                    potential = reference_heat_recovered

                predictions.append(potential)

            return np.array(predictions)

        return predictor

    def _calculate_lime_explanation(
        self,
        features: np.ndarray,
        feature_names: List[str],
        prediction_fn: Callable,
    ) -> LIMEResult:
        """Calculate LIME explanation using the lime library."""
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=self._training_data,
            feature_names=feature_names,
            mode="regression",
            discretize_continuous=self.discretize_continuous,
            kernel_width=self.kernel_width * np.sqrt(len(features)),
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
        explanation_text = []

        for feat, weight in exp.as_list():
            feature_weights[feat] = round(weight, 6)
            direction = "increases" if weight > 0 else "decreases"
            explanation_text.append(
                f"{feat}: {direction} prediction by {abs(weight):.4f}"
            )

        local_prediction = prediction_fn(features.reshape(1, -1))[0]
        intercept = exp.intercept[0] if hasattr(exp, 'intercept') else 0.0
        score = exp.score if hasattr(exp, 'score') else 0.0

        computation_hash = self._compute_hash(
            features, feature_weights, local_prediction
        )

        return LIMEResult(
            feature_names=feature_names,
            feature_weights=feature_weights,
            local_prediction=round(local_prediction, 4),
            intercept=round(intercept, 4),
            score=round(score, 4),
            num_features_used=len(feature_weights),
            explanation_text=explanation_text,
            computation_hash=computation_hash,
        )

    def _calculate_fallback_explanation(
        self,
        features: np.ndarray,
        feature_names: List[str],
        prediction_fn: Callable,
    ) -> LIMEResult:
        """Fallback: weighted linear regression for explanation."""
        # Generate perturbed samples
        n_samples = self.num_samples
        perturbations = np.random.randn(n_samples, len(features)) * 0.1
        X_perturbed = features + perturbations * np.abs(features)
        X_perturbed = np.maximum(X_perturbed, 0.01)

        # Get predictions
        y = prediction_fn(X_perturbed)

        # Compute weights (exponential kernel)
        distances = np.sqrt(np.sum(perturbations ** 2, axis=1))
        weights = np.exp(-distances ** 2 / (2 * self.kernel_width ** 2))

        # Weighted linear regression
        try:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            model.fit(X_perturbed, y, sample_weight=weights)
            coefficients = model.coef_
            intercept = model.intercept_
            score = model.score(X_perturbed, y, sample_weight=weights)
        except ImportError:
            # Manual weighted least squares
            W = np.diag(weights)
            X_aug = np.column_stack([np.ones(n_samples), X_perturbed])
            try:
                beta = np.linalg.solve(
                    X_aug.T @ W @ X_aug,
                    X_aug.T @ W @ y
                )
                intercept = beta[0]
                coefficients = beta[1:]
                y_pred = X_aug @ beta
                ss_res = np.sum(weights * (y - y_pred) ** 2)
                ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
                score = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            except np.linalg.LinAlgError:
                coefficients = np.zeros(len(features))
                intercept = 0.0
                score = 0.0

        # Build feature weights
        feature_weights = {}
        explanation_text = []

        indices = np.argsort(np.abs(coefficients))[::-1][:self.num_features]
        for i in indices:
            weight = coefficients[i]
            name = feature_names[i]
            feature_weights[name] = round(float(weight), 6)
            direction = "increases" if weight > 0 else "decreases"
            explanation_text.append(
                f"{name}: {direction} prediction by {abs(weight):.4f}"
            )

        local_prediction = prediction_fn(features.reshape(1, -1))[0]

        computation_hash = self._compute_hash(
            features, feature_weights, local_prediction
        )

        return LIMEResult(
            feature_names=feature_names,
            feature_weights=feature_weights,
            local_prediction=round(float(local_prediction), 4),
            intercept=round(float(intercept), 4),
            score=round(float(score), 4),
            num_features_used=len(feature_weights),
            explanation_text=explanation_text,
            computation_hash=computation_hash,
        )

    def _compute_hash(
        self,
        features: np.ndarray,
        weights: Dict[str, float],
        prediction: float,
    ) -> str:
        """Compute SHA-256 hash for provenance."""
        data = {
            "features": features.tolist(),
            "weights": weights,
            "prediction": prediction,
            "version": self.VERSION,
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
