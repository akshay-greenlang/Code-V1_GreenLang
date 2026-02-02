"""
SHAP-based Explainability for GL-016 Waterguard

This module provides SHAP (SHapley Additive exPlanations) based explanations
for water treatment recommendations. Optimized for tree-based models.

All explanations are deterministic and derived from structured data.
NO generative AI is used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from .explanation_schemas import (
    ExplanationMethod,
    ExplanationStabilityMetrics,
    FeatureContribution,
    FeatureDirection,
    GlobalExplanation,
    LocalExplanation,
)

logger = logging.getLogger(__name__)


@dataclass
class SHAPSummaryStatistics:
    """Summary statistics for SHAP values across a model version."""
    model_version: str
    feature_means: Dict[str, float] = field(default_factory=dict)
    feature_stds: Dict[str, float] = field(default_factory=dict)
    feature_mins: Dict[str, float] = field(default_factory=dict)
    feature_maxs: Dict[str, float] = field(default_factory=dict)
    sample_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def update(self, feature_name: str, shap_value: float) -> None:
        """Update running statistics for a feature."""
        if feature_name not in self.feature_means:
            self.feature_means[feature_name] = shap_value
            self.feature_stds[feature_name] = 0.0
            self.feature_mins[feature_name] = shap_value
            self.feature_maxs[feature_name] = shap_value
        else:
            # Welford's online algorithm for mean and variance
            old_mean = self.feature_means[feature_name]
            self.sample_count += 1
            delta = shap_value - old_mean
            self.feature_means[feature_name] += delta / self.sample_count
            delta2 = shap_value - self.feature_means[feature_name]
            self.feature_stds[feature_name] += delta * delta2

            self.feature_mins[feature_name] = min(
                self.feature_mins[feature_name], shap_value
            )
            self.feature_maxs[feature_name] = max(
                self.feature_maxs[feature_name], shap_value
            )
        self.last_updated = datetime.utcnow()

    def get_std(self, feature_name: str) -> float:
        """Get standard deviation for a feature."""
        if self.sample_count < 2:
            return 0.0
        variance = self.feature_stds.get(feature_name, 0.0) / (self.sample_count - 1)
        return np.sqrt(variance) if variance > 0 else 0.0


@dataclass
class SHAPExplanation:
    """Container for SHAP explanation results."""
    shap_values: np.ndarray
    base_value: float
    feature_names: List[str]
    feature_values: np.ndarray
    expected_value: float
    model_output: float
    computation_time_ms: float

    def to_local_explanation(
        self,
        recommendation_id: str,
        model_version: str,
        feature_units: Optional[Dict[str, str]] = None,
        feature_percentiles: Optional[Dict[str, float]] = None
    ) -> LocalExplanation:
        """Convert SHAP results to LocalExplanation schema."""
        feature_units = feature_units or {}
        feature_percentiles = feature_percentiles or {}

        contributions = []
        for i, name in enumerate(self.feature_names):
            shap_val = float(self.shap_values[i])
            feat_val = float(self.feature_values[i])

            if shap_val > 0.001:
                direction = FeatureDirection.INCREASING
            elif shap_val < -0.001:
                direction = FeatureDirection.DECREASING
            else:
                direction = FeatureDirection.NEUTRAL

            contributions.append(FeatureContribution(
                feature_name=name,
                value=feat_val,
                contribution=shap_val,
                direction=direction,
                unit=feature_units.get(name),
                percentile=feature_percentiles.get(name)
            ))

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Calculate confidence based on explanation quality
        total_contribution = sum(abs(c.contribution) for c in contributions)
        top_3_contribution = sum(
            abs(c.contribution) for c in contributions[:3]
        )
        concentration = top_3_contribution / total_contribution if total_contribution > 0 else 0
        confidence = min(0.5 + concentration * 0.5, 0.99)

        return LocalExplanation(
            recommendation_id=recommendation_id,
            method=ExplanationMethod.SHAP,
            features=contributions,
            confidence=confidence,
            base_value=self.base_value,
            prediction_value=self.model_output,
            model_version=model_version
        )


class WaterguardSHAPExplainer:
    """
    SHAP-based explainer for Waterguard ML models.

    Supports tree-based models (XGBoost, LightGBM, Random Forest, etc.)
    with optimized TreeExplainer, and falls back to KernelExplainer
    for other model types.

    Features:
    - Local explanations for individual predictions
    - Global feature importance summaries
    - Explanation stability checks
    - SHAP value caching for performance
    - Summary statistics per model version
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
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        cache_size: int = 1000,
        n_perturbations: int = 5,
        perturbation_std: float = 0.01
    ):
        """
        Initialize SHAP explainer.

        Args:
            background_data: Background dataset for SHAP (required for some explainers)
            feature_names: Names of features in order
            cache_size: Maximum number of cached explanations
            n_perturbations: Number of perturbations for stability checks
            perturbation_std: Standard deviation for perturbations
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP library is required. Install with: pip install shap"
            )

        self.background_data = background_data
        self.feature_names = feature_names or []
        self.cache_size = cache_size
        self.n_perturbations = n_perturbations
        self.perturbation_std = perturbation_std

        self._explainer = None
        self._explainer_type = None
        self._cache: Dict[str, SHAPExplanation] = {}
        self._summary_stats: Dict[str, SHAPSummaryStatistics] = {}

        # Feature units lookup
        self._feature_units = {
            name: info['unit']
            for name, info in self.WATER_FEATURES.items()
        }

    def _get_cache_key(self, input_data: np.ndarray) -> str:
        """Generate cache key from input data."""
        data_bytes = input_data.tobytes()
        return hashlib.md5(data_bytes).hexdigest()

    def _initialize_explainer(self, model: Any) -> None:
        """Initialize appropriate SHAP explainer for model type."""
        model_type = type(model).__name__

        # Try TreeExplainer first (most efficient for tree models)
        tree_model_types = [
            'XGBClassifier', 'XGBRegressor',
            'LGBMClassifier', 'LGBMRegressor',
            'RandomForestClassifier', 'RandomForestRegressor',
            'GradientBoostingClassifier', 'GradientBoostingRegressor',
            'CatBoostClassifier', 'CatBoostRegressor',
            'DecisionTreeClassifier', 'DecisionTreeRegressor'
        ]

        if model_type in tree_model_types:
            try:
                self._explainer = shap.TreeExplainer(model)
                self._explainer_type = 'tree'
                logger.info(f"Initialized TreeExplainer for {model_type}")
                return
            except Exception as e:
                logger.warning(f"TreeExplainer failed: {e}, trying alternatives")

        # Try LinearExplainer for linear models
        linear_model_types = [
            'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
            'LogisticRegression', 'SGDClassifier', 'SGDRegressor'
        ]

        if model_type in linear_model_types and self.background_data is not None:
            try:
                self._explainer = shap.LinearExplainer(
                    model, self.background_data
                )
                self._explainer_type = 'linear'
                logger.info(f"Initialized LinearExplainer for {model_type}")
                return
            except Exception as e:
                logger.warning(f"LinearExplainer failed: {e}")

        # Fall back to KernelExplainer (works for any model)
        if self.background_data is not None:
            # Use a sample of background data for efficiency
            bg_sample = self.background_data
            if len(bg_sample) > 100:
                indices = np.random.choice(len(bg_sample), 100, replace=False)
                bg_sample = bg_sample[indices]

            self._explainer = shap.KernelExplainer(
                model.predict, bg_sample
            )
            self._explainer_type = 'kernel'
            logger.info(f"Initialized KernelExplainer for {model_type}")
        else:
            raise ValueError(
                f"Cannot initialize explainer for {model_type} without background data"
            )

    def generate_local_explanation(
        self,
        model: Any,
        input_data: np.ndarray,
        recommendation: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> SHAPExplanation:
        """
        Generate SHAP-based local explanation for a prediction.

        Args:
            model: Trained ML model
            input_data: Input features (1D or 2D array)
            recommendation: Optional recommendation details
            use_cache: Whether to use cached results

        Returns:
            SHAPExplanation with SHAP values and metadata
        """
        start_time = time.time()

        # Ensure 2D input
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        # Check cache
        cache_key = self._get_cache_key(input_data)
        if use_cache and cache_key in self._cache:
            logger.debug(f"Cache hit for explanation: {cache_key[:8]}")
            return self._cache[cache_key]

        # Initialize explainer if needed
        if self._explainer is None:
            self._initialize_explainer(model)

        # Compute SHAP values
        shap_values = self._explainer.shap_values(input_data)

        # Handle multi-output models
        if isinstance(shap_values, list):
            # For classification, use positive class
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Get base value
        if hasattr(self._explainer, 'expected_value'):
            expected_value = self._explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            expected_value = 0.0

        # Flatten if needed
        shap_values = np.array(shap_values).flatten()
        feature_values = input_data.flatten()

        # Get model prediction
        try:
            model_output = float(model.predict(input_data)[0])
        except Exception:
            model_output = float(expected_value + np.sum(shap_values))

        # Use provided feature names or generate defaults
        feature_names = self.feature_names or [
            f"feature_{i}" for i in range(len(shap_values))
        ]

        computation_time = (time.time() - start_time) * 1000

        explanation = SHAPExplanation(
            shap_values=shap_values,
            base_value=float(expected_value),
            feature_names=feature_names,
            feature_values=feature_values,
            expected_value=float(expected_value),
            model_output=model_output,
            computation_time_ms=computation_time
        )

        # Update cache
        if use_cache:
            if len(self._cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = explanation

        return explanation

    def generate_global_summary(
        self,
        model: Any,
        training_data: np.ndarray,
        sample_size: Optional[int] = None
    ) -> GlobalExplanation:
        """
        Generate global feature importance summary.

        Args:
            model: Trained ML model
            training_data: Training data for computing global importance
            sample_size: Optional sample size (uses all data if None)

        Returns:
            GlobalExplanation with feature importances
        """
        start_time = time.time()

        # Sample data if needed
        if sample_size and len(training_data) > sample_size:
            indices = np.random.choice(len(training_data), sample_size, replace=False)
            data_sample = training_data[indices]
        else:
            data_sample = training_data
            sample_size = len(training_data)

        # Initialize explainer if needed
        if self._explainer is None:
            if self.background_data is None:
                self.background_data = training_data
            self._initialize_explainer(model)

        # Compute SHAP values for all samples
        shap_values = self._explainer.shap_values(data_sample)

        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        shap_values = np.array(shap_values)

        # Compute mean absolute SHAP values (feature importance)
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        feature_std = np.std(np.abs(shap_values), axis=0)

        feature_names = self.feature_names or [
            f"feature_{i}" for i in range(len(feature_importance))
        ]

        # Get model version
        model_version = getattr(model, 'version', 'unknown')
        if model_version == 'unknown':
            model_version = hashlib.md5(
                str(model.__class__.__name__).encode()
            ).hexdigest()[:8]

        computation_time = time.time() - start_time

        # Update summary statistics
        if model_version not in self._summary_stats:
            self._summary_stats[model_version] = SHAPSummaryStatistics(
                model_version=model_version
            )

        for i, name in enumerate(feature_names):
            for shap_val in shap_values[:, i]:
                self._summary_stats[model_version].update(name, float(shap_val))

        return GlobalExplanation(
            model_version=model_version,
            feature_importances=dict(zip(feature_names, feature_importance.tolist())),
            feature_importance_std=dict(zip(feature_names, feature_std.tolist())),
            sample_size=sample_size,
            computation_time_seconds=computation_time,
            metadata={
                'explainer_type': self._explainer_type,
                'total_samples_available': len(training_data)
            }
        )

    def check_explanation_stability(
        self,
        model: Any,
        input_data: np.ndarray,
        explanation: SHAPExplanation
    ) -> ExplanationStabilityMetrics:
        """
        Check stability of explanation under small perturbations.

        Args:
            model: Trained ML model
            input_data: Original input data
            explanation: SHAP explanation to check

        Returns:
            ExplanationStabilityMetrics with stability assessment
        """
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        original_shap = explanation.shap_values
        perturbation_results = []

        for _ in range(self.n_perturbations):
            # Add small Gaussian noise
            noise = np.random.normal(0, self.perturbation_std, input_data.shape)
            perturbed_input = input_data + noise * np.abs(input_data)

            # Get explanation for perturbed input
            perturbed_explanation = self.generate_local_explanation(
                model, perturbed_input, use_cache=False
            )
            perturbation_results.append(perturbed_explanation.shap_values)

        # Compute stability metrics
        perturbation_array = np.array(perturbation_results)

        # Consistency: correlation with original
        correlations = []
        for perturbed in perturbation_results:
            if np.std(original_shap) > 0 and np.std(perturbed) > 0:
                corr = np.corrcoef(original_shap, perturbed)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        perturbation_consistency = float(np.mean(correlations)) if correlations else 0.5

        # Lipschitz constant approximation
        max_change = np.max(np.abs(perturbation_array - original_shap))
        lipschitz_constant = max_change / self.perturbation_std if self.perturbation_std > 0 else 0

        # Determine stability
        passed = perturbation_consistency > 0.7 and lipschitz_constant < 10

        warnings = []
        if perturbation_consistency < 0.7:
            warnings.append(
                f"Low perturbation consistency ({perturbation_consistency:.2f})"
            )
        if lipschitz_constant > 10:
            warnings.append(
                f"High sensitivity to input changes (Lipschitz={lipschitz_constant:.2f})"
            )

        return ExplanationStabilityMetrics(
            explanation_id=str(uuid.uuid4()),
            lipschitz_constant=lipschitz_constant,
            perturbation_consistency=perturbation_consistency,
            passed_stability_check=passed,
            stability_warnings=warnings
        )

    def get_summary_statistics(
        self, model_version: str
    ) -> Optional[SHAPSummaryStatistics]:
        """Get stored summary statistics for a model version."""
        return self._summary_stats.get(model_version)

    def get_feature_units(self) -> Dict[str, str]:
        """Get feature unit mappings."""
        return self._feature_units.copy()

    def clear_cache(self) -> None:
        """Clear the explanation cache."""
        self._cache.clear()
        logger.info("SHAP explanation cache cleared")
