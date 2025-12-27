"""
LIME-based Explainability for GL-016 Waterguard

This module provides LIME (Local Interpretable Model-agnostic Explanations)
based explanations for water treatment recommendations.

All explanations are deterministic and derived from structured data.
NO generative AI is used.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime = None

from .explanation_schemas import (
    ExplanationMethod,
    ExplanationStabilityMetrics,
    FeatureContribution,
    FeatureDirection,
    LocalExplanation,
)

logger = logging.getLogger(__name__)


@dataclass
class LIMEExplanation:
    """Container for LIME explanation results."""
    feature_weights: Dict[str, float]
    feature_values: Dict[str, float]
    intercept: float
    local_prediction: float
    model_prediction: float
    r2_score: float
    feature_names: List[str]
    computation_time_ms: float

    def to_local_explanation(
        self,
        recommendation_id: str,
        model_version: str,
        feature_units: Optional[Dict[str, str]] = None,
        feature_percentiles: Optional[Dict[str, float]] = None
    ) -> LocalExplanation:
        """Convert LIME results to LocalExplanation schema."""
        feature_units = feature_units or {}
        feature_percentiles = feature_percentiles or {}

        contributions = []
        for name in self.feature_names:
            weight = self.feature_weights.get(name, 0.0)
            value = self.feature_values.get(name, 0.0)

            if weight > 0.001:
                direction = FeatureDirection.INCREASING
            elif weight < -0.001:
                direction = FeatureDirection.DECREASING
            else:
                direction = FeatureDirection.NEUTRAL

            contributions.append(FeatureContribution(
                feature_name=name,
                value=value,
                contribution=weight,
                direction=direction,
                unit=feature_units.get(name),
                percentile=feature_percentiles.get(name)
            ))

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Confidence based on R2 score and explanation quality
        base_confidence = self.r2_score * 0.5 + 0.3

        # Boost if top features explain most
        total_abs_weight = sum(abs(c.contribution) for c in contributions)
        if total_abs_weight > 0:
            top_3_weight = sum(abs(c.contribution) for c in contributions[:3])
            concentration = top_3_weight / total_abs_weight
            base_confidence += concentration * 0.2

        confidence = min(max(base_confidence, 0.1), 0.99)

        return LocalExplanation(
            recommendation_id=recommendation_id,
            method=ExplanationMethod.LIME,
            features=contributions,
            confidence=confidence,
            base_value=self.intercept,
            prediction_value=self.model_prediction,
            model_version=model_version
        )


class WaterguardLIMEExplainer:
    """
    LIME-based explainer for Waterguard ML models.

    Provides model-agnostic local explanations by fitting interpretable
    surrogate models (linear regression) to explain individual predictions.

    Features:
    - Local explanations for individual predictions
    - Configurable number of features to show
    - Human-readable feature contributions
    - Explanation caching for performance
    """

    # Water treatment feature definitions
    WATER_FEATURES = {
        'conductivity': {'unit': 'uS/cm', 'typical_range': (100, 3000)},
        'ph': {'unit': 'pH', 'typical_range': (6.5, 9.5)},
        'temperature': {'unit': 'C', 'typical_range': (10, 60)},
        'tds': {'unit': 'ppm', 'typical_range': (50, 2000)},
        'hardness': {'unit': 'ppm CaCO3', 'typical_range': (50, 500)},
        'alkalinity': {'unit': 'ppm CaCO3', 'typical_range': (50, 300)},
        'silica': {'unit': 'ppm', 'typical_range': (0, 150)},
        'chloride': {'unit': 'ppm', 'typical_range': (0, 250)},
        'sulfate': {'unit': 'ppm', 'typical_range': (0, 500)},
        'iron': {'unit': 'ppm', 'typical_range': (0, 5)},
        'cycles_of_concentration': {'unit': '', 'typical_range': (2, 10)},
        'makeup_flow': {'unit': 'm3/h', 'typical_range': (0, 1000)},
        'blowdown_rate': {'unit': '%', 'typical_range': (0.5, 10)},
    }

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        categorical_features: Optional[List[int]] = None,
        num_features: int = 10,
        num_samples: int = 5000,
        kernel_width: Optional[float] = None,
        cache_size: int = 500,
        random_state: int = 42
    ):
        """
        Initialize LIME explainer.

        Args:
            training_data: Training dataset for LIME
            feature_names: Names of features in order
            categorical_features: Indices of categorical features
            num_features: Maximum features to include in explanations
            num_samples: Number of samples for LIME perturbation
            kernel_width: Width of exponential kernel (None for auto)
            cache_size: Maximum number of cached explanations
            random_state: Random seed for reproducibility
        """
        if not LIME_AVAILABLE:
            raise ImportError(
                "LIME library is required. Install with: pip install lime"
            )

        self.training_data = training_data
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.num_features = num_features
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.cache_size = cache_size
        self.random_state = random_state

        # Initialize LIME explainer
        self._explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            categorical_features=self.categorical_features,
            mode='regression',
            kernel_width=kernel_width,
            random_state=random_state,
            verbose=False
        )

        self._cache: Dict[str, LIMEExplanation] = {}

        # Feature units lookup
        self._feature_units = {
            name: info['unit']
            for name, info in self.WATER_FEATURES.items()
        }

        # Compute training data statistics for percentile calculations
        self._training_stats = {
            name: {
                'mean': np.mean(training_data[:, i]),
                'std': np.std(training_data[:, i]),
                'min': np.min(training_data[:, i]),
                'max': np.max(training_data[:, i]),
                'p25': np.percentile(training_data[:, i], 25),
                'p50': np.percentile(training_data[:, i], 50),
                'p75': np.percentile(training_data[:, i], 75),
            }
            for i, name in enumerate(feature_names)
        }

    def _get_cache_key(self, input_data: np.ndarray) -> str:
        """Generate cache key from input data."""
        data_bytes = input_data.tobytes()
        return hashlib.md5(data_bytes).hexdigest()

    def _compute_percentile(self, feature_name: str, value: float) -> float:
        """Compute percentile of value in training distribution."""
        stats = self._training_stats.get(feature_name)
        if not stats:
            return 50.0

        # Simple percentile estimation using quartiles
        if value <= stats['min']:
            return 0.0
        elif value >= stats['max']:
            return 100.0
        elif value <= stats['p25']:
            return 25 * (value - stats['min']) / (stats['p25'] - stats['min'] + 1e-10)
        elif value <= stats['p50']:
            return 25 + 25 * (value - stats['p25']) / (stats['p50'] - stats['p25'] + 1e-10)
        elif value <= stats['p75']:
            return 50 + 25 * (value - stats['p50']) / (stats['p75'] - stats['p50'] + 1e-10)
        else:
            return 75 + 25 * (value - stats['p75']) / (stats['max'] - stats['p75'] + 1e-10)

    def generate_local_explanation(
        self,
        model: Any,
        input_data: np.ndarray,
        num_features: Optional[int] = None,
        use_cache: bool = True
    ) -> LIMEExplanation:
        """
        Generate LIME-based local explanation for a prediction.

        Args:
            model: Trained ML model with predict method
            input_data: Input features (1D array)
            num_features: Override default number of features
            use_cache: Whether to use cached results

        Returns:
            LIMEExplanation with feature weights and metadata
        """
        start_time = time.time()

        # Ensure 1D input
        if input_data.ndim > 1:
            input_data = input_data.flatten()

        # Check cache
        cache_key = self._get_cache_key(input_data)
        if use_cache and cache_key in self._cache:
            logger.debug(f"Cache hit for LIME explanation: {cache_key[:8]}")
            return self._cache[cache_key]

        # Get number of features to explain
        n_features = num_features or self.num_features
        n_features = min(n_features, len(self.feature_names))

        # Get model prediction function
        if hasattr(model, 'predict_proba'):
            # For classifiers, use probability of positive class
            def predict_fn(x):
                proba = model.predict_proba(x)
                return proba[:, 1] if proba.ndim > 1 else proba
        else:
            predict_fn = model.predict

        # Generate LIME explanation
        explanation = self._explainer.explain_instance(
            input_data,
            predict_fn,
            num_features=n_features,
            num_samples=self.num_samples
        )

        # Extract feature weights
        feature_weights = {}
        for feature, weight in explanation.as_list():
            # Parse feature name from LIME format
            # LIME uses format like "feature_name <= 1.5" or "1.0 < feature_name <= 2.0"
            parsed_name = self._parse_lime_feature(feature)
            if parsed_name:
                feature_weights[parsed_name] = weight

        # Get feature values
        feature_values = {
            name: float(input_data[i])
            for i, name in enumerate(self.feature_names)
        }

        # Get model prediction
        try:
            model_pred = float(predict_fn(input_data.reshape(1, -1))[0])
        except Exception:
            model_pred = float(explanation.local_pred[0]) if hasattr(explanation, 'local_pred') else 0.0

        # Get local prediction from LIME
        local_pred = float(explanation.local_pred[0]) if hasattr(explanation, 'local_pred') else model_pred

        # Get R2 score (how well the linear model fits locally)
        r2_score = float(explanation.score) if hasattr(explanation, 'score') else 0.5

        # Get intercept
        intercept = float(explanation.intercept[0]) if hasattr(explanation, 'intercept') else 0.0

        computation_time = (time.time() - start_time) * 1000

        lime_explanation = LIMEExplanation(
            feature_weights=feature_weights,
            feature_values=feature_values,
            intercept=intercept,
            local_prediction=local_pred,
            model_prediction=model_pred,
            r2_score=r2_score,
            feature_names=self.feature_names,
            computation_time_ms=computation_time
        )

        # Update cache
        if use_cache:
            if len(self._cache) >= self.cache_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = lime_explanation

        return lime_explanation

    def _parse_lime_feature(self, lime_feature: str) -> Optional[str]:
        """Parse feature name from LIME's formatted string."""
        for name in self.feature_names:
            if name in lime_feature:
                return name
        return None

    def generate_human_readable_explanation(
        self,
        explanation: LIMEExplanation,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Convert LIME explanation to human-readable format.

        Args:
            explanation: LIME explanation
            top_n: Number of top features to include

        Returns:
            List of human-readable feature contributions
        """
        contributions = []

        # Sort features by absolute weight
        sorted_features = sorted(
            explanation.feature_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        for feature_name, weight in sorted_features:
            value = explanation.feature_values.get(feature_name, 0.0)
            unit = self._feature_units.get(feature_name, '')
            percentile = self._compute_percentile(feature_name, value)

            # Generate human-readable description
            if weight > 0:
                effect = "increases"
                direction = "up"
            elif weight < 0:
                effect = "decreases"
                direction = "down"
            else:
                effect = "has minimal effect on"
                direction = "neutral"

            contributions.append({
                'feature': feature_name,
                'value': value,
                'unit': unit,
                'weight': weight,
                'abs_weight': abs(weight),
                'direction': direction,
                'effect': effect,
                'percentile': percentile,
                'description': f"{feature_name} = {value:.2f} {unit}".strip(),
                'impact_description': f"{feature_name} {effect} the recommendation"
            })

        return contributions

    def get_feature_units(self) -> Dict[str, str]:
        """Get feature unit mappings."""
        return self._feature_units.copy()

    def get_training_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get training data statistics."""
        return self._training_stats.copy()

    def clear_cache(self) -> None:
        """Clear the explanation cache."""
        self._cache.clear()
        logger.info("LIME explanation cache cleared")

    def set_num_samples(self, num_samples: int) -> None:
        """Update number of samples for LIME perturbations."""
        self.num_samples = num_samples
        logger.info(f"LIME num_samples set to {num_samples}")

    def set_num_features(self, num_features: int) -> None:
        """Update default number of features in explanations."""
        self.num_features = num_features
        logger.info(f"LIME num_features set to {num_features}")
