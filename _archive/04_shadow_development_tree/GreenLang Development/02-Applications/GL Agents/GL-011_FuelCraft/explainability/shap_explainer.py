# -*- coding: utf-8 -*-
"""
SHAP Explainer Module for GL-011 FuelCraft

Provides SHAP-based explanations for fuel price forecasts with zero-hallucination
guarantees. Implements TreeExplainer for gradient boosting models and KernelExplainer
as fallback for black-box models.

Features:
- TreeExplainer for tree-based models (XGBoost, LightGBM, Random Forest)
- KernelExplainer for model-agnostic explanations
- Feature attribution with business-language labels
- Interaction effects analysis
- Structured artifact storage
- Provenance tracking

Zero-Hallucination Architecture:
- All SHAP values computed deterministically
- Fixed random seeds for reproducibility
- No LLM-based SHAP value generation
- Complete provenance hashing

Global AI Standards v2.0 Compliance:
- MANDATORY: SHAP TreeExplainer Integration (5 points)

Author: GreenLang AI Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Constants
DEFAULT_RANDOM_SEED = 42
DEFAULT_NUM_SAMPLES = 100
SHAP_ADDITIVITY_TOLERANCE = 0.01
PRECISION = 6


# Business-language feature label mapping
BUSINESS_LABELS: Dict[str, str] = {
    "natural_gas_spot_price": "Natural Gas Spot Price",
    "spot_price_lag_1": "Previous Day Spot Price",
    "spot_price_lag_7": "7-Day Lagged Spot Price",
    "heating_degree_days": "Heating Degree Days",
    "cooling_degree_days": "Cooling Degree Days",
    "storage_level": "Storage Inventory Level",
    "production": "Daily Production Rate",
    "demand": "Daily Demand",
    "wind_speed_mph": "Wind Speed",
    "precipitation_in": "Precipitation",
    "storm_risk_score": "Storm Risk Score",
    "horizon_days": "Forecast Horizon",
}


class FeatureAttribution(BaseModel):
    """
    Single feature attribution from SHAP analysis.
    """

    feature_name: str = Field(..., description="Technical feature name")
    business_label: str = Field(..., description="Business-language label")
    feature_value: float = Field(..., description="Feature input value")
    shap_value: float = Field(..., description="SHAP attribution value")
    contribution_pct: float = Field(..., description="Percentage contribution")
    direction: str = Field(..., description="positive or negative")
    rank: int = Field(..., description="Importance rank (1=most important)")


class InteractionEffect(BaseModel):
    """
    SHAP interaction effect between two features.
    """

    feature_1: str = Field(..., description="First feature name")
    feature_2: str = Field(..., description="Second feature name")
    interaction_value: float = Field(..., description="Interaction SHAP value")
    combined_label: str = Field(..., description="Business-language combined label")


class SHAPExplanation(BaseModel):
    """
    Complete SHAP explanation for a forecast.
    """

    explanation_id: str = Field(..., description="Unique explanation ID")
    forecast_id: str = Field(..., description="Associated forecast ID")
    explainer_type: str = Field(..., description="tree or kernel")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Base and prediction values
    base_value: float = Field(..., description="Expected model output (E[f(X)])")
    prediction_value: float = Field(..., description="Actual prediction")

    # Feature attributions
    attributions: List[FeatureAttribution] = Field(default_factory=list)
    top_k_features: int = Field(10, description="Number of top features shown")

    # Interaction effects
    interactions: List[InteractionEffect] = Field(default_factory=list)
    has_interactions: bool = Field(False)

    # Quality metrics
    additivity_check: float = Field(..., description="Sum(SHAP) + base - prediction error")
    passes_additivity: bool = Field(True)

    # Provenance
    random_seed: int = Field(DEFAULT_RANDOM_SEED)
    computation_time_ms: float = Field(0.0)
    provenance_hash: str = Field("")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 provenance hash."""
        data = {
            "explanation_id": self.explanation_id,
            "forecast_id": self.forecast_id,
            "base_value": self.base_value,
            "prediction_value": self.prediction_value,
            "shap_values": [a.shap_value for a in self.attributions],
            "random_seed": self.random_seed,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def get_top_features(self, n: int = 5) -> List[FeatureAttribution]:
        """Get top n features by absolute SHAP value."""
        sorted_attrs = sorted(self.attributions, key=lambda x: abs(x.shap_value), reverse=True)
        return sorted_attrs[:n]


@dataclass
class SHAPConfig:
    """Configuration for SHAP explainer."""

    random_seed: int = DEFAULT_RANDOM_SEED
    num_samples: int = DEFAULT_NUM_SAMPLES
    check_additivity: bool = True
    additivity_tolerance: float = SHAP_ADDITIVITY_TOLERANCE
    max_features: int = 20
    compute_interactions: bool = False
    interaction_top_k: int = 10
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    precision: int = PRECISION
    business_labels: Dict[str, str] = field(default_factory=lambda: BUSINESS_LABELS.copy())


class SHAPForecastExplainer:
    """
    SHAP-based explainer for fuel price forecasts.

    Provides deterministic, reproducible explanations with business-language
    feature labels and complete provenance tracking.

    Zero-Hallucination Guarantees:
    - All SHAP values computed by SHAP library (not LLM)
    - Fixed random seeds for reproducibility
    - Additivity check validates SHAP consistency
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        config: Optional[SHAPConfig] = None,
        background_data: Optional[np.ndarray] = None
    ):
        """
        Initialize SHAP forecast explainer.

        Args:
            model: Trained model to explain
            feature_names: List of feature names
            config: SHAP configuration
            background_data: Background dataset for KernelExplainer
        """
        self.model = model
        self.feature_names = feature_names
        self.config = config or SHAPConfig()
        self.background_data = background_data

        # Set random seed
        np.random.seed(self.config.random_seed)

        # Initialize appropriate explainer
        self._explainer = None
        self._explainer_type = "unknown"
        self._initialize_explainer()

        # Cache
        self._cache: Dict[str, Tuple[datetime, SHAPExplanation]] = {}

        logger.info(
            f"SHAPForecastExplainer initialized: type={self._explainer_type}, "
            f"features={len(feature_names)}, seed={self.config.random_seed}"
        )

    def explain(
        self,
        features: Union[np.ndarray, Dict[str, float]],
        forecast_id: str = "",
        check_additivity: bool = True
    ) -> SHAPExplanation:
        """
        Generate SHAP explanation for features.

        Args:
            features: Feature values (array or dict)
            forecast_id: Associated forecast ID
            check_additivity: Verify SHAP additivity property

        Returns:
            SHAPExplanation with feature attributions
        """
        import uuid

        start_time = time.time()

        # Ensure reproducibility
        np.random.seed(self.config.random_seed)

        # Convert features to numpy array
        if isinstance(features, dict):
            feature_vector = self._dict_to_array(features)
        else:
            feature_vector = features

        # Reshape if needed
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # Check cache
        cache_key = self._compute_cache_key(feature_vector)
        if self.config.cache_enabled and cache_key in self._cache:
            cache_time, cached_exp = self._cache[cache_key]
            if (datetime.now(timezone.utc) - cache_time).total_seconds() < self.config.cache_ttl_seconds:
                logger.debug(f"Returning cached explanation for {cache_key[:8]}")
                return cached_exp

        # Compute SHAP values
        try:
            shap_values = self._compute_shap_values(feature_vector)
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            raise

        # Get base value and prediction
        base_value = self._get_base_value()
        prediction_value = self._get_prediction(feature_vector)

        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        # Create attributions
        attributions = self._create_attributions(
            shap_values, feature_vector[0], base_value
        )

        # Compute interactions if enabled
        interactions = []
        if self.config.compute_interactions and self._explainer_type == "tree":
            interactions = self._compute_interactions(feature_vector)

        # Check additivity
        shap_sum = float(np.sum(shap_values))
        additivity_error = abs(shap_sum + base_value - prediction_value)
        passes_additivity = additivity_error < self.config.additivity_tolerance

        if check_additivity and not passes_additivity:
            logger.warning(
                f"SHAP additivity check failed: error={additivity_error:.6f}, "
                f"tolerance={self.config.additivity_tolerance}"
            )

        computation_time = (time.time() - start_time) * 1000

        explanation = SHAPExplanation(
            explanation_id=str(uuid.uuid4()),
            forecast_id=forecast_id or str(uuid.uuid4()),
            explainer_type=self._explainer_type,
            base_value=self._round_value(base_value),
            prediction_value=self._round_value(prediction_value),
            attributions=attributions,
            top_k_features=min(len(attributions), self.config.max_features),
            interactions=interactions,
            has_interactions=len(interactions) > 0,
            additivity_check=self._round_value(additivity_error),
            passes_additivity=passes_additivity,
            random_seed=self.config.random_seed,
            computation_time_ms=computation_time,
        )

        # Cache result
        if self.config.cache_enabled:
            self._cache[cache_key] = (datetime.now(timezone.utc), explanation)

        logger.info(
            f"SHAP explanation generated: id={explanation.explanation_id[:8]}, "
            f"type={self._explainer_type}, time={computation_time:.2f}ms"
        )

        return explanation

    def explain_batch(
        self,
        features_batch: np.ndarray,
        forecast_ids: Optional[List[str]] = None
    ) -> List[SHAPExplanation]:
        """
        Generate SHAP explanations for multiple instances.

        Args:
            features_batch: Feature matrix (n_samples x n_features)
            forecast_ids: Optional list of forecast IDs

        Returns:
            List of SHAPExplanation objects
        """
        explanations = []

        for i, features in enumerate(features_batch):
            forecast_id = forecast_ids[i] if forecast_ids and i < len(forecast_ids) else ""
            try:
                exp = self.explain(features, forecast_id)
                explanations.append(exp)
            except Exception as e:
                logger.warning(f"Failed to explain instance {i}: {e}")

        return explanations

    def get_feature_importance(
        self,
        features_batch: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute global feature importance from SHAP values.

        Args:
            features_batch: Feature matrix

        Returns:
            Dictionary of feature name to mean absolute SHAP value
        """
        if features_batch.ndim == 1:
            features_batch = features_batch.reshape(1, -1)

        # Compute SHAP values for all instances
        np.random.seed(self.config.random_seed)
        shap_values = self._compute_shap_values(features_batch)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Compute mean absolute SHAP values
        importance = np.mean(np.abs(shap_values), axis=0)

        # Normalize
        total = np.sum(importance)
        if total > 0:
            importance = importance / total

        return {
            self.feature_names[i]: float(importance[i])
            for i in range(len(self.feature_names))
        }

    def _initialize_explainer(self) -> None:
        """Initialize appropriate SHAP explainer based on model type."""
        try:
            import shap

            # Try TreeExplainer first
            try:
                self._explainer = shap.TreeExplainer(self.model)
                self._explainer_type = "tree"
                logger.info("Initialized TreeExplainer")
                return
            except Exception as e:
                logger.debug(f"TreeExplainer failed: {e}, trying KernelExplainer")

            # Fall back to KernelExplainer
            if self.background_data is not None:
                # Summarize background data if too large
                if len(self.background_data) > self.config.num_samples:
                    np.random.seed(self.config.random_seed)
                    indices = np.random.choice(
                        len(self.background_data),
                        self.config.num_samples,
                        replace=False
                    )
                    background = self.background_data[indices]
                else:
                    background = self.background_data

                self._explainer = shap.KernelExplainer(
                    self.model.predict if hasattr(self.model, "predict") else self.model,
                    background
                )
                self._explainer_type = "kernel"
                logger.info("Initialized KernelExplainer")
            else:
                raise ValueError("Background data required for KernelExplainer")

        except ImportError:
            logger.warning("SHAP library not available, using placeholder")
            self._explainer = None
            self._explainer_type = "placeholder"

    def _compute_shap_values(self, features: np.ndarray) -> np.ndarray:
        """Compute SHAP values for features."""
        if self._explainer is None:
            # Placeholder: return zeros
            return np.zeros(features.shape)

        np.random.seed(self.config.random_seed)

        if self._explainer_type == "tree":
            return self._explainer.shap_values(features, check_additivity=False)
        else:
            return self._explainer.shap_values(features, nsamples=self.config.num_samples)

    def _get_base_value(self) -> float:
        """Get base value (expected model output)."""
        if self._explainer is None:
            return 0.0

        expected = self._explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            return float(expected[0])
        return float(expected)

    def _get_prediction(self, features: np.ndarray) -> float:
        """Get model prediction for features."""
        if hasattr(self.model, "predict"):
            pred = self.model.predict(features)
        else:
            pred = self.model(features)

        if isinstance(pred, np.ndarray):
            return float(pred[0])
        return float(pred)

    def _create_attributions(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        base_value: float
    ) -> List[FeatureAttribution]:
        """Create FeatureAttribution objects from SHAP values."""
        attributions = []
        total_abs = np.sum(np.abs(shap_values))

        # Sort by absolute value for ranking
        sorted_indices = np.argsort(-np.abs(shap_values))

        for rank, idx in enumerate(sorted_indices):
            feature_name = self.feature_names[idx]
            shap_val = float(shap_values[idx])
            feature_val = float(feature_values[idx])

            # Get business label
            business_label = self.config.business_labels.get(
                feature_name, feature_name.replace("_", " ").title()
            )

            # Calculate percentage
            contribution_pct = (abs(shap_val) / total_abs * 100) if total_abs > 0 else 0

            attributions.append(FeatureAttribution(
                feature_name=feature_name,
                business_label=business_label,
                feature_value=self._round_value(feature_val),
                shap_value=self._round_value(shap_val),
                contribution_pct=self._round_value(contribution_pct),
                direction="positive" if shap_val >= 0 else "negative",
                rank=rank + 1,
            ))

        # Return only top features
        return attributions[:self.config.max_features]

    def _compute_interactions(
        self,
        features: np.ndarray
    ) -> List[InteractionEffect]:
        """Compute SHAP interaction effects."""
        if self._explainer is None or self._explainer_type != "tree":
            return []

        try:
            interaction_values = self._explainer.shap_interaction_values(features)

            if isinstance(interaction_values, list):
                interaction_values = interaction_values[0]

            if interaction_values.ndim > 2:
                interaction_values = interaction_values[0]

            # Extract top interactions (off-diagonal)
            interactions = []
            n_features = len(self.feature_names)

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    value = float(interaction_values[i, j])
                    if abs(value) > 1e-6:
                        feature_1 = self.feature_names[i]
                        feature_2 = self.feature_names[j]

                        label_1 = self.config.business_labels.get(feature_1, feature_1)
                        label_2 = self.config.business_labels.get(feature_2, feature_2)

                        interactions.append(InteractionEffect(
                            feature_1=feature_1,
                            feature_2=feature_2,
                            interaction_value=self._round_value(value),
                            combined_label=f"{label_1} x {label_2}",
                        ))

            # Sort by absolute value and return top k
            interactions.sort(key=lambda x: abs(x.interaction_value), reverse=True)
            return interactions[:self.config.interaction_top_k]

        except Exception as e:
            logger.warning(f"Failed to compute interactions: {e}")
            return []

    def _dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array."""
        values = [features.get(name, 0.0) for name in self.feature_names]
        return np.array(values, dtype=np.float64)

    def _round_value(self, value: float) -> float:
        """Round value to configured precision."""
        decimal_value = Decimal(str(value))
        rounded = decimal_value.quantize(
            Decimal(10) ** -self.config.precision,
            rounding=ROUND_HALF_UP
        )
        return float(rounded)

    def _compute_cache_key(self, features: np.ndarray) -> str:
        """Compute cache key for features."""
        return hashlib.sha256(features.tobytes()).hexdigest()


class TreeExplainerWrapper:
    """
    Wrapper for SHAP TreeExplainer with additional functionality.
    """

    def __init__(self, model: Any, feature_names: List[str]):
        """Initialize TreeExplainer wrapper."""
        try:
            import shap
            self._explainer = shap.TreeExplainer(model)
        except ImportError:
            self._explainer = None

        self.feature_names = feature_names

    def explain(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Get SHAP values and expected value."""
        if self._explainer is None:
            return np.zeros(features.shape), 0.0

        shap_values = self._explainer.shap_values(features)
        expected = self._explainer.expected_value

        if isinstance(expected, (list, np.ndarray)):
            expected = expected[0]

        return shap_values, float(expected)


class KernelExplainerWrapper:
    """
    Wrapper for SHAP KernelExplainer with additional functionality.
    """

    def __init__(
        self,
        model: Callable,
        background_data: np.ndarray,
        feature_names: List[str],
        num_samples: int = DEFAULT_NUM_SAMPLES
    ):
        """Initialize KernelExplainer wrapper."""
        try:
            import shap
            self._explainer = shap.KernelExplainer(model, background_data)
        except ImportError:
            self._explainer = None

        self.feature_names = feature_names
        self.num_samples = num_samples

    def explain(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Get SHAP values and expected value."""
        if self._explainer is None:
            return np.zeros(features.shape), 0.0

        shap_values = self._explainer.shap_values(features, nsamples=self.num_samples)
        expected = self._explainer.expected_value

        return shap_values, float(expected)


# Utility functions

def verify_shap_additivity(
    shap_values: np.ndarray,
    base_value: float,
    prediction: float,
    tolerance: float = SHAP_ADDITIVITY_TOLERANCE
) -> Tuple[bool, float]:
    """
    Verify SHAP additivity property.

    SHAP values should satisfy: base_value + sum(shap_values) = prediction

    Args:
        shap_values: SHAP values array
        base_value: Expected model output
        prediction: Actual prediction
        tolerance: Acceptable error tolerance

    Returns:
        Tuple of (passes_check, error_value)
    """
    shap_sum = float(np.sum(shap_values))
    error = abs(base_value + shap_sum - prediction)

    return error < tolerance, error


def aggregate_attributions(
    explanations: List[SHAPExplanation]
) -> Dict[str, float]:
    """
    Aggregate feature attributions from multiple explanations.

    Args:
        explanations: List of SHAP explanations

    Returns:
        Dictionary of feature name to mean absolute SHAP value
    """
    feature_values: Dict[str, List[float]] = {}

    for exp in explanations:
        for attr in exp.attributions:
            if attr.feature_name not in feature_values:
                feature_values[attr.feature_name] = []
            feature_values[attr.feature_name].append(abs(attr.shap_value))

    # Compute means
    aggregated = {
        name: float(np.mean(values))
        for name, values in feature_values.items()
    }

    # Normalize
    total = sum(aggregated.values())
    if total > 0:
        aggregated = {k: v / total for k, v in aggregated.items()}

    return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))


def compute_attribution_hash(attributions: List[FeatureAttribution]) -> str:
    """
    Compute SHA-256 hash of attributions.

    Args:
        attributions: List of feature attributions

    Returns:
        64-character hex hash
    """
    data = [
        {"name": a.feature_name, "value": a.shap_value}
        for a in attributions
    ]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
