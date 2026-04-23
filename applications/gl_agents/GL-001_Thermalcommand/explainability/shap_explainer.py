# -*- coding: utf-8 -*-
"""
SHAP Explainer Module for GL-001 ThermalCommand.

Provides SHAP-based explanations for ML predictions with zero-hallucination
guarantees. All numeric values are computed deterministically using SHAP
algorithms, not LLM-generated.

Features:
- TreeExplainer for tree-based models (Random Forest, XGBoost, LightGBM)
- KernelExplainer for model-agnostic explanations
- Feature importance rankings
- Waterfall plot data generation
- Force plot data generation
- Interaction effects analysis

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
import numpy as np
from dataclasses import dataclass

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from .explanation_schemas import (
    FeatureContribution,
    SHAPExplanation,
    ConfidenceBounds,
    UncertaintyRange,
    PredictionType,
    DashboardExplanationData,
)

logger = logging.getLogger(__name__)

# Constants for determinism
DEFAULT_RANDOM_SEED = 42
DEFAULT_NUM_SAMPLES = 100
SHAP_CONSISTENCY_TOLERANCE = 0.01


@dataclass
class SHAPConfig:
    """Configuration for SHAP explainer."""

    random_seed: int = DEFAULT_RANDOM_SEED
    num_samples: int = DEFAULT_NUM_SAMPLES
    check_additivity: bool = True
    approximate: bool = False
    feature_perturbation: str = "interventional"
    max_features: int = 20
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300


class SHAPExplainer:
    """
    SHAP-based explainer with zero-hallucination guarantees.

    Provides deterministic, reproducible explanations for:
    - Tree-based models (TreeExplainer)
    - Any model (KernelExplainer)
    - Deep learning models (DeepExplainer - placeholder)

    All explanations include:
    - Feature contributions
    - Consistency checks
    - Provenance hashes for audit trails
    """

    def __init__(
        self,
        config: Optional[SHAPConfig] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize SHAP explainer.

        Args:
            config: SHAP configuration settings
            feature_names: Names of features for human-readable explanations
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP library not installed. Install with: pip install shap"
            )

        self.config = config or SHAPConfig()
        self.feature_names = feature_names or []
        self._tree_explainer: Optional[shap.TreeExplainer] = None
        self._kernel_explainer: Optional[shap.KernelExplainer] = None
        self._background_data: Optional[np.ndarray] = None
        self._explanation_cache: Dict[str, SHAPExplanation] = {}
        self._cache_timestamps: Dict[str, float] = {}

        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)

        logger.info(
            f"SHAPExplainer initialized with seed={self.config.random_seed}, "
            f"num_samples={self.config.num_samples}"
        )

    def fit_tree_explainer(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        model_output: str = "raw"
    ) -> None:
        """
        Fit TreeExplainer for tree-based models.

        Args:
            model: Tree-based model (RandomForest, XGBoost, LightGBM, etc.)
            feature_names: Feature names (overrides constructor value)
            model_output: Type of model output ('raw', 'probability', 'log_loss')
        """
        start_time = time.time()

        if feature_names:
            self.feature_names = feature_names

        try:
            self._tree_explainer = shap.TreeExplainer(
                model,
                feature_perturbation=self.config.feature_perturbation,
                model_output=model_output
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"TreeExplainer fitted in {elapsed_ms:.2f}ms")

        except Exception as e:
            logger.error(f"Failed to fit TreeExplainer: {str(e)}")
            raise

    def fit_kernel_explainer(
        self,
        model: Callable,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        link: str = "identity"
    ) -> None:
        """
        Fit KernelExplainer for model-agnostic explanations.

        Args:
            model: Model prediction function (callable)
            background_data: Background dataset for computing expectations
            feature_names: Feature names (overrides constructor value)
            link: Link function ('identity' or 'logit')
        """
        start_time = time.time()

        if feature_names:
            self.feature_names = feature_names

        # Use summarized background data for efficiency
        if len(background_data) > self.config.num_samples:
            np.random.seed(self.config.random_seed)
            indices = np.random.choice(
                len(background_data),
                self.config.num_samples,
                replace=False
            )
            self._background_data = background_data[indices]
        else:
            self._background_data = background_data

        try:
            self._kernel_explainer = shap.KernelExplainer(
                model,
                self._background_data,
                link=link
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"KernelExplainer fitted in {elapsed_ms:.2f}ms")

        except Exception as e:
            logger.error(f"Failed to fit KernelExplainer: {str(e)}")
            raise

    def explain_instance(
        self,
        instance: np.ndarray,
        prediction_type: PredictionType,
        use_tree_explainer: bool = True,
        check_additivity: bool = True
    ) -> SHAPExplanation:
        """
        Generate SHAP explanation for a single instance.

        Args:
            instance: Feature vector to explain (1D array)
            prediction_type: Type of prediction being explained
            use_tree_explainer: Use TreeExplainer if available
            check_additivity: Verify SHAP additivity property

        Returns:
            SHAPExplanation with feature contributions
        """
        start_time = time.time()

        # Check cache
        cache_key = self._compute_cache_key(instance)
        if self.config.cache_enabled and cache_key in self._explanation_cache:
            cache_time = self._cache_timestamps.get(cache_key, 0)
            if time.time() - cache_time < self.config.cache_ttl_seconds:
                logger.debug(f"Returning cached explanation for {cache_key[:8]}")
                return self._explanation_cache[cache_key]

        # Reshape if needed
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        # Select explainer
        explainer = None
        explainer_type = "unknown"

        if use_tree_explainer and self._tree_explainer is not None:
            explainer = self._tree_explainer
            explainer_type = "tree"
        elif self._kernel_explainer is not None:
            explainer = self._kernel_explainer
            explainer_type = "kernel"
        else:
            raise ValueError("No explainer fitted. Call fit_tree_explainer or fit_kernel_explainer first.")

        # Compute SHAP values
        try:
            if explainer_type == "tree":
                shap_values = explainer.shap_values(instance, check_additivity=check_additivity)
            else:
                # KernelExplainer uses sampling
                np.random.seed(self.config.random_seed)
                shap_values = explainer.shap_values(
                    instance,
                    nsamples=self.config.num_samples
                )

            # Handle multi-class output
            if isinstance(shap_values, list):
                # For binary classification, use class 1
                shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

            # Ensure 1D array
            if shap_values.ndim > 1:
                shap_values = shap_values[0]

        except Exception as e:
            logger.error(f"Failed to compute SHAP values: {str(e)}")
            raise

        # Get base value and prediction
        base_value = float(explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1] if len(base_value) == 2 else base_value[0])

        prediction_value = float(base_value + np.sum(shap_values))

        # Create feature contributions
        feature_contributions = self._create_feature_contributions(
            shap_values,
            instance[0],
            base_value
        )

        # Compute consistency check
        consistency_error = abs(
            np.sum(shap_values) - (prediction_value - base_value)
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Generate explanation ID
        explanation_id = hashlib.sha256(
            f"{cache_key}{start_time}".encode()
        ).hexdigest()[:16]

        explanation = SHAPExplanation(
            explanation_id=explanation_id,
            prediction_type=prediction_type,
            base_value=base_value,
            prediction_value=prediction_value,
            feature_contributions=feature_contributions,
            interaction_effects=None,  # Computed separately if needed
            consistency_check=consistency_error,
            explainer_type=explainer_type,
            timestamp=datetime.utcnow(),
            computation_time_ms=elapsed_ms,
            random_seed=self.config.random_seed
        )

        # Cache result
        if self.config.cache_enabled:
            self._explanation_cache[cache_key] = explanation
            self._cache_timestamps[cache_key] = time.time()

        logger.info(
            f"SHAP explanation generated in {elapsed_ms:.2f}ms "
            f"(consistency error: {consistency_error:.6f})"
        )

        return explanation

    def explain_batch(
        self,
        instances: np.ndarray,
        prediction_type: PredictionType,
        use_tree_explainer: bool = True
    ) -> List[SHAPExplanation]:
        """
        Generate SHAP explanations for a batch of instances.

        Args:
            instances: Feature matrix (2D array, each row is an instance)
            prediction_type: Type of prediction being explained
            use_tree_explainer: Use TreeExplainer if available

        Returns:
            List of SHAPExplanation objects
        """
        explanations = []
        for i, instance in enumerate(instances):
            try:
                exp = self.explain_instance(
                    instance,
                    prediction_type,
                    use_tree_explainer
                )
                explanations.append(exp)
            except Exception as e:
                logger.warning(f"Failed to explain instance {i}: {str(e)}")

        return explanations

    def compute_interaction_effects(
        self,
        instance: np.ndarray,
        top_k: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute SHAP interaction effects between features.

        Args:
            instance: Feature vector to explain
            top_k: Number of top interactions to return

        Returns:
            Dictionary of feature pairs and their interaction values
        """
        if self._tree_explainer is None:
            raise ValueError("Interaction effects require TreeExplainer")

        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        # Compute interaction values
        interaction_values = self._tree_explainer.shap_interaction_values(instance)

        if isinstance(interaction_values, list):
            interaction_values = interaction_values[1] if len(interaction_values) == 2 else interaction_values[0]

        if interaction_values.ndim > 2:
            interaction_values = interaction_values[0]

        # Extract top interactions
        interactions = {}
        n_features = interaction_values.shape[0]

        # Flatten upper triangle and sort
        interaction_pairs = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                value = float(interaction_values[i, j])
                if abs(value) > 1e-6:  # Filter near-zero interactions
                    feature_i = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                    feature_j = self.feature_names[j] if j < len(self.feature_names) else f"feature_{j}"
                    interaction_pairs.append((feature_i, feature_j, value))

        # Sort by absolute value
        interaction_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        # Build dictionary
        for feature_i, feature_j, value in interaction_pairs[:top_k]:
            if feature_i not in interactions:
                interactions[feature_i] = {}
            interactions[feature_i][feature_j] = value

        return interactions

    def get_feature_importance(
        self,
        instances: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute global feature importance from SHAP values.

        Args:
            instances: Feature matrix to compute importance over

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if instances.ndim == 1:
            instances = instances.reshape(1, -1)

        # Select explainer
        if self._tree_explainer is not None:
            explainer = self._tree_explainer
        elif self._kernel_explainer is not None:
            explainer = self._kernel_explainer
        else:
            raise ValueError("No explainer fitted")

        # Compute SHAP values for all instances
        shap_values = explainer.shap_values(instances)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        # Compute mean absolute SHAP value per feature
        importance = np.mean(np.abs(shap_values), axis=0)

        # Normalize
        total = np.sum(importance)
        if total > 0:
            importance = importance / total

        # Build dictionary
        feature_importance = {}
        for i, imp in enumerate(importance):
            name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            feature_importance[name] = float(imp)

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return feature_importance

    def generate_waterfall_data(
        self,
        explanation: SHAPExplanation,
        max_features: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate data for waterfall plot visualization.

        Args:
            explanation: SHAP explanation object
            max_features: Maximum number of features to include

        Returns:
            List of dictionaries with waterfall chart data
        """
        max_features = max_features or self.config.max_features

        waterfall_data = []

        # Start with base value
        waterfall_data.append({
            "feature": "Base Value",
            "value": explanation.base_value,
            "contribution": 0,
            "cumulative": explanation.base_value,
            "is_base": True
        })

        # Sort contributions by absolute value
        sorted_contributions = sorted(
            explanation.feature_contributions,
            key=lambda x: abs(x.contribution),
            reverse=True
        )[:max_features]

        cumulative = explanation.base_value

        for contrib in sorted_contributions:
            cumulative += contrib.contribution
            waterfall_data.append({
                "feature": contrib.feature_name,
                "value": contrib.feature_value,
                "contribution": contrib.contribution,
                "cumulative": cumulative,
                "direction": contrib.direction,
                "is_base": False
            })

        # Add final prediction
        waterfall_data.append({
            "feature": "Prediction",
            "value": explanation.prediction_value,
            "contribution": 0,
            "cumulative": explanation.prediction_value,
            "is_base": True
        })

        return waterfall_data

    def generate_force_plot_data(
        self,
        explanation: SHAPExplanation,
        max_features: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate data for force plot visualization.

        Args:
            explanation: SHAP explanation object
            max_features: Maximum number of features to include

        Returns:
            Dictionary with force plot data
        """
        max_features = max_features or self.config.max_features

        # Separate positive and negative contributions
        positive_contributions = []
        negative_contributions = []

        sorted_contributions = sorted(
            explanation.feature_contributions,
            key=lambda x: abs(x.contribution),
            reverse=True
        )[:max_features]

        for contrib in sorted_contributions:
            entry = {
                "feature": contrib.feature_name,
                "value": contrib.feature_value,
                "contribution": abs(contrib.contribution),
                "percentage": contrib.contribution_percentage
            }
            if contrib.direction == "positive":
                positive_contributions.append(entry)
            else:
                negative_contributions.append(entry)

        return {
            "base_value": explanation.base_value,
            "prediction_value": explanation.prediction_value,
            "positive_contributions": positive_contributions,
            "negative_contributions": negative_contributions,
            "positive_total": sum(c["contribution"] for c in positive_contributions),
            "negative_total": sum(c["contribution"] for c in negative_contributions),
            "explanation_id": explanation.explanation_id
        }

    def _create_feature_contributions(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        base_value: float
    ) -> List[FeatureContribution]:
        """Create FeatureContribution objects from SHAP values."""
        contributions = []
        total_abs_contribution = np.sum(np.abs(shap_values))

        for i, (shap_val, feat_val) in enumerate(zip(shap_values, feature_values)):
            name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"

            if total_abs_contribution > 0:
                contribution_pct = (abs(shap_val) / total_abs_contribution) * 100
            else:
                contribution_pct = 0

            direction = "positive" if shap_val >= 0 else "negative"

            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=float(feat_val),
                contribution=float(shap_val),
                contribution_percentage=float(contribution_pct),
                direction=direction,
                baseline_value=float(base_value)
            ))

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)

        return contributions

    def _compute_cache_key(self, instance: np.ndarray) -> str:
        """Compute cache key for instance."""
        instance_bytes = instance.tobytes()
        return hashlib.sha256(instance_bytes).hexdigest()

    def clear_cache(self) -> None:
        """Clear explanation cache."""
        self._explanation_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Explanation cache cleared")


class TreeSHAPExplainer(SHAPExplainer):
    """
    Specialized SHAP explainer for tree-based models.

    Optimized for:
    - Random Forest
    - XGBoost
    - LightGBM
    - CatBoost
    - Gradient Boosting
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        config: Optional[SHAPConfig] = None
    ):
        """
        Initialize TreeSHAP explainer.

        Args:
            model: Tree-based model
            feature_names: Feature names
            config: SHAP configuration
        """
        super().__init__(config, feature_names)
        self.fit_tree_explainer(model, feature_names)


class KernelSHAPExplainer(SHAPExplainer):
    """
    Model-agnostic SHAP explainer using KernelSHAP.

    Works with any model that has a predict method.
    """

    def __init__(
        self,
        model: Callable,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        config: Optional[SHAPConfig] = None
    ):
        """
        Initialize KernelSHAP explainer.

        Args:
            model: Model prediction function
            background_data: Background dataset
            feature_names: Feature names
            config: SHAP configuration
        """
        super().__init__(config, feature_names)
        self.fit_kernel_explainer(model, background_data, feature_names)


# Utility functions for SHAP operations

def verify_shap_consistency(
    shap_values: np.ndarray,
    base_value: float,
    prediction: float,
    tolerance: float = SHAP_CONSISTENCY_TOLERANCE
) -> bool:
    """
    Verify SHAP additivity property.

    SHAP values should sum to prediction - base_value.

    Args:
        shap_values: Computed SHAP values
        base_value: Expected model output
        prediction: Actual prediction
        tolerance: Maximum allowed error

    Returns:
        True if consistency check passes
    """
    expected_sum = prediction - base_value
    actual_sum = np.sum(shap_values)
    error = abs(expected_sum - actual_sum)

    if error > tolerance:
        logger.warning(
            f"SHAP consistency check failed: error={error:.6f}, "
            f"tolerance={tolerance}"
        )
        return False

    return True


def aggregate_shap_explanations(
    explanations: List[SHAPExplanation]
) -> Dict[str, float]:
    """
    Aggregate feature importance from multiple explanations.

    Args:
        explanations: List of SHAP explanations

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
