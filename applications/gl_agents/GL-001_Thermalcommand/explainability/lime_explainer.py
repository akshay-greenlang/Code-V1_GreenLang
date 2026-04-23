# -*- coding: utf-8 -*-
"""
LIME Explainer Module for GL-001 ThermalCommand.

Provides LIME-based explanations for ML predictions with zero-hallucination
guarantees. Uses Local Interpretable Model-agnostic Explanations to create
interpretable surrogate models around individual predictions.

Features:
- LimeTabularExplainer for numeric predictions
- Local surrogate model generation
- Feature contribution extraction
- Confidence intervals for explanations
- Deterministic explanations via seed control

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass

# LIME imports
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime = None

from .explanation_schemas import (
    FeatureContribution,
    LIMEExplanation,
    ConfidenceBounds,
    UncertaintyRange,
    PredictionType,
)

logger = logging.getLogger(__name__)

# Constants for determinism
DEFAULT_RANDOM_SEED = 42
DEFAULT_NUM_SAMPLES = 5000
DEFAULT_NUM_FEATURES = 10
MIN_LOCAL_R2 = 0.7  # Minimum acceptable local fidelity


@dataclass
class LIMEConfig:
    """Configuration for LIME explainer."""

    random_seed: int = DEFAULT_RANDOM_SEED
    num_samples: int = DEFAULT_NUM_SAMPLES
    num_features: int = DEFAULT_NUM_FEATURES
    kernel_width: Optional[float] = None  # Auto-computed if None
    mode: str = "regression"  # "regression" or "classification"
    discretize_continuous: bool = True
    discretizer: str = "quartile"  # "quartile", "decile", or "entropy"
    sample_around_instance: bool = True
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300


class LIMEExplainer:
    """
    LIME-based explainer with zero-hallucination guarantees.

    Provides local interpretable explanations by:
    1. Generating perturbed samples around the instance
    2. Getting model predictions for perturbed samples
    3. Fitting a local linear model weighted by proximity
    4. Extracting feature contributions from the linear model

    All explanations are deterministic and reproducible when
    the same random seed is used.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        config: Optional[LIMEConfig] = None,
        categorical_features: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize LIME explainer.

        Args:
            training_data: Training data for computing statistics
            feature_names: Names of features
            config: LIME configuration settings
            categorical_features: Indices of categorical features
            class_names: Names of classes (for classification)
        """
        if not LIME_AVAILABLE:
            raise ImportError(
                "LIME library not installed. Install with: pip install lime"
            )

        self.config = config or LIMEConfig()
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.class_names = class_names
        self.training_data = training_data

        # Compute kernel width if not specified
        if self.config.kernel_width is None:
            self.config.kernel_width = np.sqrt(training_data.shape[1]) * 0.75

        # Initialize LIME explainer
        self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            categorical_features=categorical_features,
            class_names=class_names,
            mode=self.config.mode,
            discretize_continuous=self.config.discretize_continuous,
            discretizer=self.config.discretizer,
            sample_around_instance=self.config.sample_around_instance,
            random_state=self.config.random_seed,
            kernel_width=self.config.kernel_width
        )

        # Cache for explanations
        self._explanation_cache: Dict[str, LIMEExplanation] = {}
        self._cache_timestamps: Dict[str, float] = {}

        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)

        logger.info(
            f"LIMEExplainer initialized with {len(feature_names)} features, "
            f"seed={self.config.random_seed}, num_samples={self.config.num_samples}"
        )

    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: Callable,
        prediction_type: PredictionType,
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None
    ) -> LIMEExplanation:
        """
        Generate LIME explanation for a single instance.

        Args:
            instance: Feature vector to explain (1D array)
            predict_fn: Model prediction function
            prediction_type: Type of prediction being explained
            num_features: Number of features to include in explanation
            num_samples: Number of perturbed samples to generate

        Returns:
            LIMEExplanation with feature contributions and local model metrics
        """
        start_time = time.time()

        num_features = num_features or self.config.num_features
        num_samples = num_samples or self.config.num_samples

        # Check cache
        cache_key = self._compute_cache_key(instance, num_features, num_samples)
        if self.config.cache_enabled and cache_key in self._explanation_cache:
            cache_time = self._cache_timestamps.get(cache_key, 0)
            if time.time() - cache_time < self.config.cache_ttl_seconds:
                logger.debug(f"Returning cached LIME explanation for {cache_key[:8]}")
                return self._explanation_cache[cache_key]

        # Ensure 1D array
        if instance.ndim > 1:
            instance = instance.flatten()

        # Set random state for reproducibility
        np.random.seed(self.config.random_seed)

        try:
            # Generate LIME explanation
            lime_exp = self._lime_explainer.explain_instance(
                instance,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples
            )
        except Exception as e:
            logger.error(f"Failed to generate LIME explanation: {str(e)}")
            raise

        # Extract explanation data
        explanation_map = dict(lime_exp.as_list())

        # Get local model metrics
        local_model = lime_exp.local_exp
        local_model_r2 = lime_exp.score if hasattr(lime_exp, 'score') else 0.0
        local_model_intercept = lime_exp.intercept[1] if self.config.mode == "classification" else lime_exp.intercept

        # Get prediction value
        prediction_value = float(predict_fn(instance.reshape(1, -1))[0])
        if isinstance(prediction_value, np.ndarray):
            prediction_value = float(prediction_value[0])

        # Create feature contributions
        feature_contributions = self._create_feature_contributions(
            lime_exp,
            instance,
            prediction_value,
            local_model_intercept
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Generate explanation ID
        explanation_id = hashlib.sha256(
            f"{cache_key}{start_time}".encode()
        ).hexdigest()[:16]

        explanation = LIMEExplanation(
            explanation_id=explanation_id,
            prediction_type=prediction_type,
            prediction_value=prediction_value,
            feature_contributions=feature_contributions,
            local_model_r2=float(local_model_r2),
            local_model_intercept=float(local_model_intercept),
            neighborhood_size=num_samples,
            kernel_width=float(self.config.kernel_width),
            timestamp=datetime.utcnow(),
            computation_time_ms=elapsed_ms,
            random_seed=self.config.random_seed
        )

        # Warn if local fidelity is low
        if local_model_r2 < MIN_LOCAL_R2:
            logger.warning(
                f"Low local fidelity (R2={local_model_r2:.3f}). "
                f"Explanation may not be reliable."
            )

        # Cache result
        if self.config.cache_enabled:
            self._explanation_cache[cache_key] = explanation
            self._cache_timestamps[cache_key] = time.time()

        logger.info(
            f"LIME explanation generated in {elapsed_ms:.2f}ms "
            f"(local R2: {local_model_r2:.3f})"
        )

        return explanation

    def explain_batch(
        self,
        instances: np.ndarray,
        predict_fn: Callable,
        prediction_type: PredictionType,
        num_features: Optional[int] = None
    ) -> List[LIMEExplanation]:
        """
        Generate LIME explanations for a batch of instances.

        Args:
            instances: Feature matrix (2D array, each row is an instance)
            predict_fn: Model prediction function
            prediction_type: Type of prediction being explained
            num_features: Number of features to include

        Returns:
            List of LIMEExplanation objects
        """
        explanations = []
        for i, instance in enumerate(instances):
            try:
                exp = self.explain_instance(
                    instance,
                    predict_fn,
                    prediction_type,
                    num_features
                )
                explanations.append(exp)
            except Exception as e:
                logger.warning(f"Failed to explain instance {i}: {str(e)}")

        return explanations

    def compute_confidence_intervals(
        self,
        instance: np.ndarray,
        predict_fn: Callable,
        num_bootstrap: int = 100,
        confidence_level: float = 0.95
    ) -> Dict[str, ConfidenceBounds]:
        """
        Compute confidence intervals for feature contributions using bootstrap.

        Args:
            instance: Feature vector to explain
            predict_fn: Model prediction function
            num_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Dictionary mapping feature names to confidence bounds
        """
        # Ensure 1D array
        if instance.ndim > 1:
            instance = instance.flatten()

        # Collect bootstrap samples
        contribution_samples: Dict[str, List[float]] = {
            name: [] for name in self.feature_names
        }

        for i in range(num_bootstrap):
            # Use different seed for each bootstrap
            seed = self.config.random_seed + i
            np.random.seed(seed)

            try:
                lime_exp = self._lime_explainer.explain_instance(
                    instance,
                    predict_fn,
                    num_features=len(self.feature_names),
                    num_samples=self.config.num_samples // 2  # Faster for bootstrap
                )

                # Extract contributions
                for feature_idx, weight in lime_exp.local_exp[1] if self.config.mode == "classification" else lime_exp.local_exp[0]:
                    feature_name = self.feature_names[feature_idx]
                    contribution_samples[feature_name].append(weight)

            except Exception as e:
                logger.debug(f"Bootstrap sample {i} failed: {str(e)}")
                continue

        # Compute confidence intervals
        confidence_intervals = {}
        alpha = 1 - confidence_level

        for name, samples in contribution_samples.items():
            if len(samples) < 10:
                continue

            samples_arr = np.array(samples)
            lower = float(np.percentile(samples_arr, alpha / 2 * 100))
            upper = float(np.percentile(samples_arr, (1 - alpha / 2) * 100))

            confidence_intervals[name] = ConfidenceBounds(
                lower_bound=lower,
                upper_bound=upper,
                confidence_level=confidence_level,
                method="bootstrap"
            )

        # Reset seed
        np.random.seed(self.config.random_seed)

        return confidence_intervals

    def get_local_model(
        self,
        instance: np.ndarray,
        predict_fn: Callable
    ) -> Tuple[Dict[str, float], float, float]:
        """
        Get the local linear model coefficients.

        Args:
            instance: Feature vector to explain
            predict_fn: Model prediction function

        Returns:
            Tuple of (coefficients dict, intercept, R-squared)
        """
        if instance.ndim > 1:
            instance = instance.flatten()

        np.random.seed(self.config.random_seed)

        lime_exp = self._lime_explainer.explain_instance(
            instance,
            predict_fn,
            num_features=len(self.feature_names),
            num_samples=self.config.num_samples
        )

        # Extract coefficients
        coefficients = {}
        exp_data = lime_exp.local_exp[1] if self.config.mode == "classification" else lime_exp.local_exp[0]

        for feature_idx, weight in exp_data:
            feature_name = self.feature_names[feature_idx]
            coefficients[feature_name] = float(weight)

        intercept = lime_exp.intercept[1] if self.config.mode == "classification" else lime_exp.intercept
        r2 = lime_exp.score if hasattr(lime_exp, 'score') else 0.0

        return coefficients, float(intercept), float(r2)

    def generate_surrogate_model_data(
        self,
        instance: np.ndarray,
        predict_fn: Callable,
        num_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate data about the local surrogate model.

        Useful for understanding and visualizing the local approximation.

        Args:
            instance: Feature vector to explain
            predict_fn: Model prediction function
            num_samples: Number of samples to include

        Returns:
            Dictionary with surrogate model details
        """
        if instance.ndim > 1:
            instance = instance.flatten()

        num_samples = num_samples or self.config.num_samples

        np.random.seed(self.config.random_seed)

        lime_exp = self._lime_explainer.explain_instance(
            instance,
            predict_fn,
            num_features=len(self.feature_names),
            num_samples=num_samples
        )

        # Get coefficients and intercept
        coefficients, intercept, r2 = self.get_local_model(instance, predict_fn)

        # Get original prediction
        original_pred = float(predict_fn(instance.reshape(1, -1))[0])

        # Calculate surrogate prediction
        surrogate_pred = intercept
        for feature_name, coef in coefficients.items():
            feature_idx = self.feature_names.index(feature_name)
            surrogate_pred += coef * instance[feature_idx]

        return {
            "coefficients": coefficients,
            "intercept": intercept,
            "r_squared": r2,
            "original_prediction": original_pred,
            "surrogate_prediction": float(surrogate_pred),
            "prediction_error": abs(original_pred - surrogate_pred),
            "kernel_width": self.config.kernel_width,
            "num_samples": num_samples,
            "feature_names": self.feature_names,
            "instance_values": {
                name: float(instance[i])
                for i, name in enumerate(self.feature_names)
            }
        }

    def _create_feature_contributions(
        self,
        lime_exp,
        instance: np.ndarray,
        prediction_value: float,
        intercept: float
    ) -> List[FeatureContribution]:
        """Create FeatureContribution objects from LIME explanation."""
        contributions = []

        # Get explanation data
        if self.config.mode == "classification":
            exp_data = lime_exp.local_exp.get(1, lime_exp.local_exp.get(0, []))
        else:
            exp_data = lime_exp.local_exp.get(0, [])

        # Calculate total absolute contribution for percentages
        total_abs_contribution = sum(abs(weight) for _, weight in exp_data)

        for feature_idx, weight in exp_data:
            feature_name = self.feature_names[feature_idx]
            feature_value = float(instance[feature_idx])

            if total_abs_contribution > 0:
                contribution_pct = (abs(weight) / total_abs_contribution) * 100
            else:
                contribution_pct = 0

            direction = "positive" if weight >= 0 else "negative"

            contributions.append(FeatureContribution(
                feature_name=feature_name,
                feature_value=feature_value,
                contribution=float(weight),
                contribution_percentage=float(contribution_pct),
                direction=direction,
                baseline_value=float(intercept)
            ))

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)

        return contributions

    def _compute_cache_key(
        self,
        instance: np.ndarray,
        num_features: int,
        num_samples: int
    ) -> str:
        """Compute cache key for instance."""
        instance_bytes = instance.tobytes()
        key_str = f"{instance_bytes}{num_features}{num_samples}{self.config.random_seed}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear explanation cache."""
        self._explanation_cache.clear()
        self._cache_timestamps.clear()
        logger.info("LIME explanation cache cleared")


class TabularLIMEExplainer(LIMEExplainer):
    """
    Convenience class for tabular data LIME explanations.

    Pre-configured for typical tabular/numeric data scenarios.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        categorical_features: Optional[List[int]] = None,
        random_seed: int = DEFAULT_RANDOM_SEED
    ):
        """
        Initialize tabular LIME explainer with sensible defaults.

        Args:
            training_data: Training data for computing statistics
            feature_names: Names of features
            categorical_features: Indices of categorical features
            random_seed: Random seed for reproducibility
        """
        config = LIMEConfig(
            random_seed=random_seed,
            num_samples=5000,
            num_features=min(10, len(feature_names)),
            mode="regression",
            discretize_continuous=True,
            discretizer="quartile"
        )

        super().__init__(
            training_data=training_data,
            feature_names=feature_names,
            config=config,
            categorical_features=categorical_features
        )


# Utility functions for LIME operations

def aggregate_lime_explanations(
    explanations: List[LIMEExplanation]
) -> Dict[str, float]:
    """
    Aggregate feature importance from multiple LIME explanations.

    Args:
        explanations: List of LIME explanations

    Returns:
        Dictionary of feature names to mean absolute contributions
    """
    feature_contributions: Dict[str, List[float]] = {}

    for exp in explanations:
        for contrib in exp.feature_contributions:
            if contrib.feature_name not in feature_contributions:
                feature_contributions[contrib.feature_name] = []
            feature_contributions[contrib.feature_name].append(abs(contrib.contribution))

    # Compute mean
    aggregated = {
        name: float(np.mean(values))
        for name, values in feature_contributions.items()
    }

    # Normalize
    total = sum(aggregated.values())
    if total > 0:
        aggregated = {k: v / total for k, v in aggregated.items()}

    return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))


def compare_lime_explanations(
    exp1: LIMEExplanation,
    exp2: LIMEExplanation
) -> Dict[str, Any]:
    """
    Compare two LIME explanations.

    Useful for understanding how explanations change over time
    or between different instances.

    Args:
        exp1: First LIME explanation
        exp2: Second LIME explanation

    Returns:
        Comparison results
    """
    # Build contribution maps
    contrib1 = {c.feature_name: c.contribution for c in exp1.feature_contributions}
    contrib2 = {c.feature_name: c.contribution for c in exp2.feature_contributions}

    # Find common features
    common_features = set(contrib1.keys()) & set(contrib2.keys())

    # Calculate differences
    differences = {}
    for feature in common_features:
        diff = contrib2[feature] - contrib1[feature]
        differences[feature] = {
            "exp1_contribution": contrib1[feature],
            "exp2_contribution": contrib2[feature],
            "difference": diff,
            "relative_change": diff / contrib1[feature] if contrib1[feature] != 0 else float('inf')
        }

    return {
        "prediction_change": exp2.prediction_value - exp1.prediction_value,
        "r2_change": exp2.local_model_r2 - exp1.local_model_r2,
        "feature_differences": differences,
        "features_only_in_exp1": list(set(contrib1.keys()) - common_features),
        "features_only_in_exp2": list(set(contrib2.keys()) - common_features)
    }


def validate_lime_explanation(
    explanation: LIMEExplanation,
    min_r2: float = MIN_LOCAL_R2
) -> Tuple[bool, List[str]]:
    """
    Validate a LIME explanation for quality.

    Args:
        explanation: LIME explanation to validate
        min_r2: Minimum acceptable local R-squared

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Check local fidelity
    if explanation.local_model_r2 < min_r2:
        issues.append(
            f"Low local fidelity: R2={explanation.local_model_r2:.3f} < {min_r2}"
        )

    # Check for empty contributions
    if not explanation.feature_contributions:
        issues.append("No feature contributions in explanation")

    # Check contribution percentages sum
    total_pct = sum(c.contribution_percentage for c in explanation.feature_contributions)
    if abs(total_pct - 100) > 5:  # Allow 5% tolerance
        issues.append(
            f"Contribution percentages don't sum to 100%: {total_pct:.1f}%"
        )

    # Check for reasonable kernel width
    if explanation.kernel_width <= 0:
        issues.append(f"Invalid kernel width: {explanation.kernel_width}")

    return len(issues) == 0, issues
