"""
GreenLang Framework - LIME Explainer Module

Provides LIME-based explanations for ML predictions with zero-hallucination
guarantees. Uses Local Interpretable Model-agnostic Explanations to create
interpretable surrogate models around individual predictions.

Features:
- TabularExplainer for numeric and categorical data
- Local linear surrogate model generation
- Feature contribution extraction with confidence intervals
- Deterministic explanations via seed control
- Caching for performance optimization
- Batch explanation support

Theory:
LIME explains predictions by:
1. Generating perturbed samples around the instance
2. Getting model predictions for perturbed samples
3. Weighting samples by proximity (kernel function)
4. Fitting a local linear model to weighted samples
5. Extracting feature contributions from linear coefficients

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
import hashlib
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
import json

import numpy as np

# LIME imports with fallback
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
    ExplainerType,
)

logger = logging.getLogger(__name__)


# Configuration constants
DEFAULT_RANDOM_SEED = 42
DEFAULT_NUM_SAMPLES = 5000
DEFAULT_NUM_FEATURES = 10
DEFAULT_KERNEL_WIDTH = None  # Auto-computed if None
DEFAULT_CACHE_TTL_SECONDS = 300
MIN_LOCAL_R2 = 0.7  # Minimum acceptable local fidelity


@dataclass
class LIMEConfig:
    """
    Configuration for LIME explainer.

    Attributes:
        random_seed: Seed for reproducibility
        num_samples: Number of perturbed samples to generate
        num_features: Number of features to include in explanations
        kernel_width: Kernel width for proximity weighting (auto if None)
        mode: Prediction mode ('regression' or 'classification')
        discretize_continuous: Whether to discretize continuous features
        discretizer: Discretization method ('quartile', 'decile', 'entropy')
        sample_around_instance: Sample around instance vs training data
        cache_enabled: Enable explanation caching
        cache_ttl_seconds: Cache time-to-live in seconds
        batch_size: Batch size for batch explanations
    """
    random_seed: int = DEFAULT_RANDOM_SEED
    num_samples: int = DEFAULT_NUM_SAMPLES
    num_features: int = DEFAULT_NUM_FEATURES
    kernel_width: Optional[float] = DEFAULT_KERNEL_WIDTH
    mode: str = "regression"
    discretize_continuous: bool = True
    discretizer: str = "quartile"
    sample_around_instance: bool = True
    cache_enabled: bool = True
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS
    batch_size: int = 50


@dataclass
class CacheEntry:
    """Cache entry for LIME explanations."""
    explanation: LIMEExplanation
    timestamp: float
    access_count: int = 0


class LIMEExplainerService:
    """
    LIME-based explainer service with zero-hallucination guarantees.

    Provides local interpretable explanations by fitting sparse linear
    models in the neighborhood of each prediction. All explanations
    are deterministic and reproducible when the same random seed is used.

    Key metrics:
    - local_model_r2: How well the linear model fits locally (higher is better)
    - kernel_width: Controls the locality of the explanation

    Example:
        >>> config = LIMEConfig(random_seed=42, num_samples=5000)
        >>> explainer = LIMEExplainerService(
        ...     training_data=X_train,
        ...     feature_names=["temp", "pressure", "flow"],
        ...     config=config
        ... )
        >>> explanation = explainer.explain_instance(
        ...     instance=X_test[0],
        ...     predict_fn=model.predict,
        ...     prediction_type=PredictionType.REGRESSION
        ... )
        >>> print(f"Local R2: {explanation.local_model_r2:.3f}")
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        config: Optional[LIMEConfig] = None,
        categorical_features: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
        agent_id: str = "GL-FRAMEWORK",
        agent_version: str = "1.0.0"
    ):
        """
        Initialize LIME explainer service.

        Args:
            training_data: Training data for computing statistics
            feature_names: Names of features
            config: LIME configuration settings
            categorical_features: Indices of categorical features
            class_names: Names of classes (for classification)
            agent_id: Agent identifier for provenance tracking
            agent_version: Agent version for provenance tracking

        Raises:
            ImportError: If LIME library is not installed
            ValueError: If training_data or feature_names are invalid
        """
        if not LIME_AVAILABLE:
            raise ImportError(
                "LIME library not installed. Install with: pip install lime"
            )

        if training_data.ndim != 2:
            raise ValueError("training_data must be a 2D array")

        if len(feature_names) != training_data.shape[1]:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match "
                f"training_data columns ({training_data.shape[1]})"
            )

        self.config = config or LIMEConfig()
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.class_names = class_names
        self.training_data = training_data
        self.agent_id = agent_id
        self.agent_version = agent_version

        # Compute kernel width if not specified
        if self.config.kernel_width is None:
            self.config.kernel_width = float(np.sqrt(training_data.shape[1]) * 0.75)
            logger.debug(f"Auto-computed kernel width: {self.config.kernel_width:.4f}")

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

        # Caching
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)

        logger.info(
            f"LIMEExplainerService initialized: agent={agent_id}, "
            f"features={len(feature_names)}, seed={self.config.random_seed}, "
            f"num_samples={self.config.num_samples}"
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

        Raises:
            ValueError: If instance is invalid
            RuntimeError: If explanation generation fails
        """
        start_time = time.time()

        num_features = num_features or self.config.num_features
        num_samples = num_samples or self.config.num_samples

        # Validate instance
        instance = self._validate_instance(instance)

        # Check cache
        cache_key = self._compute_cache_key(instance, num_features, num_samples)
        if self.config.cache_enabled:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                self._cache_hits += 1
                logger.debug(f"Cache hit for key {cache_key[:8]}")
                return cached
            self._cache_misses += 1

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
            logger.error(f"Failed to generate LIME explanation: {str(e)}", exc_info=True)
            raise RuntimeError(f"LIME explanation failed: {str(e)}") from e

        # Get local model metrics
        local_model_r2 = float(lime_exp.score) if hasattr(lime_exp, 'score') else 0.0
        local_model_intercept = self._get_intercept(lime_exp)

        # Get prediction value
        prediction_value = self._get_prediction(instance, predict_fn)

        # Create feature contributions
        feature_contributions = self._create_feature_contributions(
            lime_exp,
            instance,
            prediction_value,
            local_model_intercept
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Generate explanation ID
        explanation_id = self._generate_explanation_id(cache_key, start_time)

        explanation = LIMEExplanation(
            explanation_id=explanation_id,
            prediction_type=prediction_type,
            prediction_value=prediction_value,
            feature_contributions=feature_contributions,
            local_model_r2=local_model_r2,
            local_model_intercept=float(local_model_intercept),
            neighborhood_size=num_samples,
            kernel_width=float(self.config.kernel_width),
            timestamp=datetime.now(timezone.utc),
            computation_time_ms=elapsed_ms,
            random_seed=self.config.random_seed
        )

        # Warn if local fidelity is low
        if local_model_r2 < MIN_LOCAL_R2:
            logger.warning(
                f"Low local fidelity (R2={local_model_r2:.3f}). "
                f"Explanation may not be reliable. Consider increasing num_samples."
            )

        # Cache result
        if self.config.cache_enabled:
            self._add_to_cache(cache_key, explanation)

        logger.info(
            f"LIME explanation generated: id={explanation_id[:8]}, "
            f"time={elapsed_ms:.2f}ms, local_R2={local_model_r2:.3f}"
        )

        return explanation

    def explain_batch(
        self,
        instances: np.ndarray,
        predict_fn: Callable,
        prediction_type: PredictionType,
        num_features: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[LIMEExplanation]:
        """
        Generate LIME explanations for a batch of instances.

        Args:
            instances: Feature matrix (2D array, each row is an instance)
            predict_fn: Model prediction function
            prediction_type: Type of prediction being explained
            num_features: Number of features to include
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of LIMEExplanation objects

        Raises:
            ValueError: If instances is not 2D
        """
        start_time = time.time()

        if instances.ndim != 2:
            raise ValueError("instances must be a 2D array")

        explanations = []
        total = len(instances)
        batch_size = self.config.batch_size

        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch = instances[i:batch_end]

            for j, instance in enumerate(batch):
                try:
                    exp = self.explain_instance(
                        instance,
                        predict_fn,
                        prediction_type,
                        num_features
                    )
                    explanations.append(exp)

                    if progress_callback:
                        progress_callback(i + j + 1, total)

                except Exception as e:
                    logger.warning(f"Failed to explain instance {i + j}: {str(e)}")

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Batch explanation complete: {len(explanations)}/{total} succeeded, "
            f"total_time={elapsed_ms:.2f}ms"
        )

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
        instance = self._validate_instance(instance)

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
                exp_data = self._get_local_exp_data(lime_exp)
                for feature_idx, weight in exp_data:
                    if feature_idx < len(self.feature_names):
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
        instance = self._validate_instance(instance)

        np.random.seed(self.config.random_seed)

        lime_exp = self._lime_explainer.explain_instance(
            instance,
            predict_fn,
            num_features=len(self.feature_names),
            num_samples=self.config.num_samples
        )

        # Extract coefficients
        coefficients = {}
        exp_data = self._get_local_exp_data(lime_exp)

        for feature_idx, weight in exp_data:
            if feature_idx < len(self.feature_names):
                feature_name = self.feature_names[feature_idx]
                coefficients[feature_name] = float(weight)

        intercept = self._get_intercept(lime_exp)
        r2 = float(lime_exp.score) if hasattr(lime_exp, 'score') else 0.0

        return coefficients, float(intercept), r2

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
        instance = self._validate_instance(instance)
        num_samples = num_samples or self.config.num_samples

        # Get local model
        coefficients, intercept, r2 = self.get_local_model(instance, predict_fn)

        # Get original prediction
        original_pred = self._get_prediction(instance, predict_fn)

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

    def validate_explanation(
        self,
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
        total_pct = sum(
            c.contribution_percentage for c in explanation.feature_contributions
        )
        if abs(total_pct - 100) > 5:  # Allow 5% tolerance
            issues.append(
                f"Contribution percentages don't sum to 100%: {total_pct:.1f}%"
            )

        # Check for reasonable kernel width
        if explanation.kernel_width <= 0:
            issues.append(f"Invalid kernel width: {explanation.kernel_width}")

        return len(issues) == 0, issues

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0.0
            )
        }

    def clear_cache(self) -> None:
        """Clear explanation cache."""
        self._cache.clear()
        logger.info("LIME explanation cache cleared")

    # Private methods

    def _validate_instance(self, instance: np.ndarray) -> np.ndarray:
        """Validate and flatten instance to 1D."""
        if instance.ndim > 1:
            instance = instance.flatten()

        if len(instance) != len(self.feature_names):
            raise ValueError(
                f"Instance has {len(instance)} features, "
                f"expected {len(self.feature_names)}"
            )

        return instance

    def _get_prediction(
        self,
        instance: np.ndarray,
        predict_fn: Callable
    ) -> float:
        """Get model prediction for instance."""
        prediction = predict_fn(instance.reshape(1, -1))
        if isinstance(prediction, np.ndarray):
            prediction = prediction.flatten()[0]
        return float(prediction)

    def _get_intercept(self, lime_exp) -> float:
        """Extract intercept from LIME explanation."""
        if self.config.mode == "classification":
            return float(lime_exp.intercept[1]) if hasattr(lime_exp, 'intercept') else 0.0
        else:
            return float(lime_exp.intercept) if hasattr(lime_exp, 'intercept') else 0.0

    def _get_local_exp_data(self, lime_exp) -> List[Tuple[int, float]]:
        """Extract local explanation data from LIME explanation."""
        if self.config.mode == "classification":
            return lime_exp.local_exp.get(1, lime_exp.local_exp.get(0, []))
        else:
            return lime_exp.local_exp.get(0, [])

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
        exp_data = self._get_local_exp_data(lime_exp)

        # Calculate total absolute contribution for percentages
        total_abs_contribution = sum(abs(weight) for _, weight in exp_data)

        for feature_idx, weight in exp_data:
            if feature_idx >= len(self.feature_names):
                continue

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
        key_data = {
            "instance": instance.tobytes().hex(),
            "num_features": num_features,
            "num_samples": num_samples,
            "seed": self.config.random_seed
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _generate_explanation_id(
        self,
        cache_key: str,
        timestamp: float
    ) -> str:
        """Generate unique explanation ID."""
        id_data = f"{cache_key}{timestamp}{uuid.uuid4()}"
        return hashlib.sha256(id_data.encode()).hexdigest()[:16]

    def _get_from_cache(self, key: str) -> Optional[LIMEExplanation]:
        """Get explanation from cache if valid."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if time.time() - entry.timestamp > self.config.cache_ttl_seconds:
            del self._cache[key]
            return None

        entry.access_count += 1
        return entry.explanation

    def _add_to_cache(self, key: str, explanation: LIMEExplanation) -> None:
        """Add explanation to cache."""
        self._cache[key] = CacheEntry(
            explanation=explanation,
            timestamp=time.time()
        )


# Convenience classes

class TabularLIMEExplainer(LIMEExplainerService):
    """
    Convenience class for tabular data LIME explanations.

    Pre-configured for typical tabular/numeric data scenarios
    with sensible defaults.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        categorical_features: Optional[List[int]] = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        agent_id: str = "GL-FRAMEWORK",
        agent_version: str = "1.0.0"
    ):
        """
        Initialize tabular LIME explainer with sensible defaults.

        Args:
            training_data: Training data for computing statistics
            feature_names: Names of features
            categorical_features: Indices of categorical features
            random_seed: Random seed for reproducibility
            agent_id: Agent identifier
            agent_version: Agent version
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
            categorical_features=categorical_features,
            agent_id=agent_id,
            agent_version=agent_version
        )


class ClassificationLIMEExplainer(LIMEExplainerService):
    """
    LIME explainer configured for classification tasks.

    Optimized for binary and multi-class classification with
    probability outputs.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
        categorical_features: Optional[List[int]] = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        agent_id: str = "GL-FRAMEWORK",
        agent_version: str = "1.0.0"
    ):
        """
        Initialize classification LIME explainer.

        Args:
            training_data: Training data for computing statistics
            feature_names: Names of features
            class_names: Names of classes
            categorical_features: Indices of categorical features
            random_seed: Random seed for reproducibility
            agent_id: Agent identifier
            agent_version: Agent version
        """
        config = LIMEConfig(
            random_seed=random_seed,
            num_samples=5000,
            num_features=min(10, len(feature_names)),
            mode="classification",
            discretize_continuous=True,
            discretizer="quartile"
        )

        super().__init__(
            training_data=training_data,
            feature_names=feature_names,
            config=config,
            categorical_features=categorical_features,
            class_names=class_names,
            agent_id=agent_id,
            agent_version=agent_version
        )


# Utility functions

def aggregate_lime_explanations(
    explanations: List[LIMEExplanation],
    normalize: bool = True
) -> Dict[str, float]:
    """
    Aggregate feature importance from multiple LIME explanations.

    Args:
        explanations: List of LIME explanations
        normalize: Normalize to sum to 1.0

    Returns:
        Dictionary of feature names to mean absolute contributions
    """
    feature_contributions: Dict[str, List[float]] = {}

    for exp in explanations:
        for contrib in exp.feature_contributions:
            if contrib.feature_name not in feature_contributions:
                feature_contributions[contrib.feature_name] = []
            feature_contributions[contrib.feature_name].append(
                abs(contrib.contribution)
            )

    # Compute mean
    aggregated = {
        name: float(np.mean(values))
        for name, values in feature_contributions.items()
    }

    # Normalize if requested
    if normalize:
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
        Comparison results with feature-level differences
    """
    # Build contribution maps
    contrib1 = {c.feature_name: c.contribution for c in exp1.feature_contributions}
    contrib2 = {c.feature_name: c.contribution for c in exp2.feature_contributions}

    # Find all features
    all_features = set(contrib1.keys()) | set(contrib2.keys())

    # Calculate differences
    differences = {}
    for feature in all_features:
        val1 = contrib1.get(feature, 0.0)
        val2 = contrib2.get(feature, 0.0)
        diff = val2 - val1

        differences[feature] = {
            "exp1_contribution": val1,
            "exp2_contribution": val2,
            "difference": diff,
            "relative_change": (
                diff / val1 if val1 != 0 else float('inf') if diff != 0 else 0.0
            )
        }

    return {
        "prediction_change": exp2.prediction_value - exp1.prediction_value,
        "r2_change": exp2.local_model_r2 - exp1.local_model_r2,
        "feature_differences": differences,
        "features_only_in_exp1": list(set(contrib1.keys()) - set(contrib2.keys())),
        "features_only_in_exp2": list(set(contrib2.keys()) - set(contrib1.keys())),
        "exp1_id": exp1.explanation_id,
        "exp2_id": exp2.explanation_id
    }


def compute_explanation_stability(
    instance: np.ndarray,
    predict_fn: Callable,
    lime_explainer: LIMEExplainerService,
    num_runs: int = 10
) -> Dict[str, UncertaintyRange]:
    """
    Compute stability of LIME explanations across multiple runs.

    Args:
        instance: Feature vector to explain
        predict_fn: Model prediction function
        lime_explainer: LIME explainer service
        num_runs: Number of explanation runs

    Returns:
        Dictionary of feature names to uncertainty ranges
    """
    contribution_samples: Dict[str, List[float]] = {}

    original_seed = lime_explainer.config.random_seed

    for i in range(num_runs):
        # Use different seeds
        lime_explainer.config.random_seed = original_seed + i
        np.random.seed(lime_explainer.config.random_seed)

        try:
            exp = lime_explainer.explain_instance(
                instance,
                predict_fn,
                PredictionType.REGRESSION
            )

            for contrib in exp.feature_contributions:
                if contrib.feature_name not in contribution_samples:
                    contribution_samples[contrib.feature_name] = []
                contribution_samples[contrib.feature_name].append(contrib.contribution)

        except Exception as e:
            logger.warning(f"Stability run {i} failed: {str(e)}")

    # Restore original seed
    lime_explainer.config.random_seed = original_seed

    # Compute uncertainty ranges
    uncertainty_ranges = {}
    for name, samples in contribution_samples.items():
        if len(samples) < 2:
            continue

        samples_arr = np.array(samples)
        uncertainty_ranges[name] = UncertaintyRange(
            mean=float(np.mean(samples_arr)),
            std=float(np.std(samples_arr)),
            min_value=float(np.min(samples_arr)),
            max_value=float(np.max(samples_arr)),
            percentile_5=float(np.percentile(samples_arr, 5)),
            percentile_95=float(np.percentile(samples_arr, 95)),
            num_samples=len(samples)
        )

    return uncertainty_ranges
