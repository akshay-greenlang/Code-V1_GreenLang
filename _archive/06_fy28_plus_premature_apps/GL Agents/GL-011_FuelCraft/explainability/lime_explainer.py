# -*- coding: utf-8 -*-
"""
LIME Explainer Module for GL-011 FuelCraft

Provides LIME-based local interpretable explanations for fuel price forecasts.
Implements LimeTabularExplainer with validation against SHAP for consistency.

Features:
- LimeTabularExplainer for numeric predictions
- Local surrogate model extraction
- Feature importance rankings
- Comparison with SHAP for validation
- Business-language feature labels

Zero-Hallucination Architecture:
- All LIME values computed deterministically
- Fixed random seeds for reproducibility
- No LLM-based explanation generation
- Complete provenance hashing

Global AI Standards v2.0 Compliance:
- RECOMMENDED: LIME Explainer (3 points)

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
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Constants
DEFAULT_RANDOM_SEED = 42
DEFAULT_NUM_SAMPLES = 5000
DEFAULT_NUM_FEATURES = 10
MIN_LOCAL_FIDELITY = 0.7
PRECISION = 6

# Reuse business labels from SHAP module
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


class LocalSurrogateModel(BaseModel):
    """
    Local surrogate model fitted by LIME.
    """

    coefficients: Dict[str, float] = Field(default_factory=dict)
    intercept: float = Field(0.0)
    r_squared: float = Field(0.0, ge=0.0, le=1.0)
    kernel_width: float = Field(0.0)


class LIMEExplanation(BaseModel):
    """
    Complete LIME explanation for a forecast.
    """

    explanation_id: str = Field(..., description="Unique explanation ID")
    forecast_id: str = Field(..., description="Associated forecast ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Prediction info
    prediction_value: float = Field(..., description="Model prediction")

    # Feature contributions (from local model)
    feature_weights: Dict[str, float] = Field(default_factory=dict)
    feature_labels: Dict[str, str] = Field(default_factory=dict)
    feature_values: Dict[str, float] = Field(default_factory=dict)
    top_features: List[str] = Field(default_factory=list)

    # Local model info
    local_model: LocalSurrogateModel = Field(default_factory=LocalSurrogateModel)

    # Quality metrics
    local_fidelity_r2: float = Field(..., ge=0.0, le=1.0, description="Local model R-squared")
    is_reliable: bool = Field(True, description="Explanation meets fidelity threshold")

    # Configuration
    num_samples: int = Field(DEFAULT_NUM_SAMPLES)
    num_features: int = Field(DEFAULT_NUM_FEATURES)

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
            "prediction_value": self.prediction_value,
            "feature_weights": self.feature_weights,
            "random_seed": self.random_seed,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def get_weight(self, feature_name: str) -> Optional[float]:
        """Get weight for a specific feature."""
        return self.feature_weights.get(feature_name)


@dataclass
class LIMEConfig:
    """Configuration for LIME explainer."""

    random_seed: int = DEFAULT_RANDOM_SEED
    num_samples: int = DEFAULT_NUM_SAMPLES
    num_features: int = DEFAULT_NUM_FEATURES
    kernel_width: Optional[float] = None  # Auto-computed if None
    mode: str = "regression"
    discretize_continuous: bool = True
    discretizer: str = "quartile"
    sample_around_instance: bool = True
    min_local_fidelity: float = MIN_LOCAL_FIDELITY
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    precision: int = PRECISION
    business_labels: Dict[str, str] = field(default_factory=lambda: BUSINESS_LABELS.copy())


class LIMEForecastExplainer:
    """
    LIME-based explainer for fuel price forecasts.

    Provides local interpretable explanations by fitting a
    linear model around individual predictions.

    Zero-Hallucination Guarantees:
    - All LIME values computed by LIME library (not LLM)
    - Fixed random seeds for reproducibility
    - Local fidelity check validates explanation quality
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        config: Optional[LIMEConfig] = None,
        categorical_features: Optional[List[int]] = None
    ):
        """
        Initialize LIME forecast explainer.

        Args:
            training_data: Training data for computing statistics
            feature_names: List of feature names
            config: LIME configuration
            categorical_features: Indices of categorical features
        """
        self.training_data = training_data
        self.feature_names = feature_names
        self.config = config or LIMEConfig()
        self.categorical_features = categorical_features or []

        # Compute kernel width if not specified
        if self.config.kernel_width is None:
            self.config.kernel_width = np.sqrt(training_data.shape[1]) * 0.75

        # Set random seed
        np.random.seed(self.config.random_seed)

        # Initialize LIME explainer
        self._explainer = None
        self._initialize_explainer()

        # Cache
        self._cache: Dict[str, Tuple[datetime, LIMEExplanation]] = {}

        logger.info(
            f"LIMEForecastExplainer initialized: features={len(feature_names)}, "
            f"samples={self.config.num_samples}, seed={self.config.random_seed}"
        )

    def explain(
        self,
        features: Union[np.ndarray, Dict[str, float]],
        predict_fn: Callable,
        forecast_id: str = "",
        num_features: Optional[int] = None
    ) -> LIMEExplanation:
        """
        Generate LIME explanation for features.

        Args:
            features: Feature values (array or dict)
            predict_fn: Model prediction function
            forecast_id: Associated forecast ID
            num_features: Number of features to include

        Returns:
            LIMEExplanation with local model
        """
        import uuid

        start_time = time.time()
        num_features = num_features or self.config.num_features

        # Ensure reproducibility
        np.random.seed(self.config.random_seed)

        # Convert features to numpy array
        if isinstance(features, dict):
            feature_vector = self._dict_to_array(features)
        else:
            feature_vector = features

        # Flatten if needed
        if feature_vector.ndim > 1:
            feature_vector = feature_vector.flatten()

        # Check cache
        cache_key = self._compute_cache_key(feature_vector, num_features)
        if self.config.cache_enabled and cache_key in self._cache:
            cache_time, cached_exp = self._cache[cache_key]
            if (datetime.now(timezone.utc) - cache_time).total_seconds() < self.config.cache_ttl_seconds:
                logger.debug(f"Returning cached LIME explanation for {cache_key[:8]}")
                return cached_exp

        # Generate LIME explanation
        try:
            lime_exp = self._generate_explanation(feature_vector, predict_fn, num_features)
        except Exception as e:
            logger.error(f"LIME computation failed: {e}")
            raise

        # Get prediction
        prediction_value = float(predict_fn(feature_vector.reshape(1, -1))[0])

        # Extract feature weights
        feature_weights, local_model = self._extract_weights(lime_exp)

        # Build feature labels and values
        feature_labels = {
            name: self.config.business_labels.get(name, name.replace("_", " ").title())
            for name in feature_weights.keys()
        }

        feature_values = {
            name: self._round_value(feature_vector[self.feature_names.index(name)])
            for name in feature_weights.keys()
            if name in self.feature_names
        }

        # Get top features by absolute weight
        sorted_features = sorted(
            feature_weights.keys(),
            key=lambda k: abs(feature_weights[k]),
            reverse=True
        )

        # Check fidelity
        is_reliable = local_model.r_squared >= self.config.min_local_fidelity

        if not is_reliable:
            logger.warning(
                f"LIME local fidelity below threshold: "
                f"R2={local_model.r_squared:.3f} < {self.config.min_local_fidelity}"
            )

        computation_time = (time.time() - start_time) * 1000

        explanation = LIMEExplanation(
            explanation_id=str(uuid.uuid4()),
            forecast_id=forecast_id or str(uuid.uuid4()),
            prediction_value=self._round_value(prediction_value),
            feature_weights=feature_weights,
            feature_labels=feature_labels,
            feature_values=feature_values,
            top_features=sorted_features[:num_features],
            local_model=local_model,
            local_fidelity_r2=local_model.r_squared,
            is_reliable=is_reliable,
            num_samples=self.config.num_samples,
            num_features=num_features,
            random_seed=self.config.random_seed,
            computation_time_ms=computation_time,
        )

        # Cache result
        if self.config.cache_enabled:
            self._cache[cache_key] = (datetime.now(timezone.utc), explanation)

        logger.info(
            f"LIME explanation generated: id={explanation.explanation_id[:8]}, "
            f"R2={local_model.r_squared:.3f}, time={computation_time:.2f}ms"
        )

        return explanation

    def explain_batch(
        self,
        features_batch: np.ndarray,
        predict_fn: Callable,
        forecast_ids: Optional[List[str]] = None
    ) -> List[LIMEExplanation]:
        """
        Generate LIME explanations for multiple instances.

        Args:
            features_batch: Feature matrix
            predict_fn: Model prediction function
            forecast_ids: Optional list of forecast IDs

        Returns:
            List of LIMEExplanation objects
        """
        explanations = []

        for i, features in enumerate(features_batch):
            forecast_id = forecast_ids[i] if forecast_ids and i < len(forecast_ids) else ""
            try:
                exp = self.explain(features, predict_fn, forecast_id)
                explanations.append(exp)
            except Exception as e:
                logger.warning(f"Failed to explain instance {i}: {e}")

        return explanations

    def _initialize_explainer(self) -> None:
        """Initialize LIME explainer."""
        try:
            import lime
            import lime.lime_tabular

            self._explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.training_data,
                feature_names=self.feature_names,
                categorical_features=self.categorical_features,
                mode=self.config.mode,
                discretize_continuous=self.config.discretize_continuous,
                discretizer=self.config.discretizer,
                sample_around_instance=self.config.sample_around_instance,
                random_state=self.config.random_seed,
                kernel_width=self.config.kernel_width,
            )
            logger.info("LIME TabularExplainer initialized")

        except ImportError:
            logger.warning("LIME library not available")
            self._explainer = None

    def _generate_explanation(
        self,
        features: np.ndarray,
        predict_fn: Callable,
        num_features: int
    ) -> Any:
        """Generate LIME explanation."""
        if self._explainer is None:
            raise RuntimeError("LIME explainer not initialized")

        np.random.seed(self.config.random_seed)

        return self._explainer.explain_instance(
            features,
            predict_fn,
            num_features=num_features,
            num_samples=self.config.num_samples,
        )

    def _extract_weights(
        self,
        lime_exp: Any
    ) -> Tuple[Dict[str, float], LocalSurrogateModel]:
        """Extract feature weights and local model from LIME explanation."""
        # Get explanation as list
        exp_list = lime_exp.as_list()

        # Parse feature weights
        feature_weights = {}
        for feature_desc, weight in exp_list:
            # LIME returns feature descriptions like "feature_name <= 5.0"
            # Extract the base feature name
            for name in self.feature_names:
                if name in feature_desc:
                    if name not in feature_weights:
                        feature_weights[name] = 0.0
                    feature_weights[name] += float(weight)
                    break

        # Round weights
        feature_weights = {
            k: self._round_value(v) for k, v in feature_weights.items()
        }

        # Get local model info
        r_squared = lime_exp.score if hasattr(lime_exp, 'score') else 0.0
        intercept = lime_exp.intercept[0] if hasattr(lime_exp, 'intercept') else 0.0

        local_model = LocalSurrogateModel(
            coefficients=feature_weights,
            intercept=float(intercept),
            r_squared=float(r_squared),
            kernel_width=float(self.config.kernel_width),
        )

        return feature_weights, local_model

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

    def _compute_cache_key(
        self,
        features: np.ndarray,
        num_features: int
    ) -> str:
        """Compute cache key for features."""
        key_data = f"{features.tobytes()}{num_features}{self.config.random_seed}"
        return hashlib.sha256(key_data.encode()).hexdigest()


class TabularExplainerWrapper:
    """
    Wrapper for LIME TabularExplainer with simplified interface.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        random_seed: int = DEFAULT_RANDOM_SEED
    ):
        """Initialize TabularExplainer wrapper."""
        config = LIMEConfig(
            random_seed=random_seed,
            num_samples=5000,
            num_features=min(10, len(feature_names)),
        )

        self._explainer = LIMEForecastExplainer(
            training_data=training_data,
            feature_names=feature_names,
            config=config,
        )

    def explain(
        self,
        features: np.ndarray,
        predict_fn: Callable
    ) -> LIMEExplanation:
        """Generate explanation for features."""
        return self._explainer.explain(features, predict_fn)


# Utility functions

def validate_lime_fidelity(
    explanation: LIMEExplanation,
    min_r2: float = MIN_LOCAL_FIDELITY
) -> Tuple[bool, List[str]]:
    """
    Validate LIME explanation fidelity.

    Args:
        explanation: LIME explanation
        min_r2: Minimum acceptable R-squared

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    if explanation.local_fidelity_r2 < min_r2:
        issues.append(
            f"Local fidelity below threshold: "
            f"R2={explanation.local_fidelity_r2:.3f} < {min_r2}"
        )

    if not explanation.feature_weights:
        issues.append("No feature weights in explanation")

    if len(explanation.top_features) == 0:
        issues.append("No top features identified")

    return len(issues) == 0, issues


def compare_shap_lime(
    shap_weights: Dict[str, float],
    lime_weights: Dict[str, float],
    tolerance: float = 0.2
) -> Dict[str, Any]:
    """
    Compare SHAP and LIME feature importances.

    Args:
        shap_weights: SHAP feature attributions (normalized)
        lime_weights: LIME feature weights (normalized)
        tolerance: Acceptable difference threshold

    Returns:
        Comparison results
    """
    # Normalize both to sum to 1
    shap_total = sum(abs(v) for v in shap_weights.values())
    lime_total = sum(abs(v) for v in lime_weights.values())

    if shap_total == 0 or lime_total == 0:
        return {"comparable": False, "reason": "Zero total weight"}

    shap_norm = {k: abs(v) / shap_total for k, v in shap_weights.items()}
    lime_norm = {k: abs(v) / lime_total for k, v in lime_weights.items()}

    # Compare common features
    common_features = set(shap_norm.keys()) & set(lime_norm.keys())
    differences = {}
    agreements = 0

    for feature in common_features:
        diff = abs(shap_norm[feature] - lime_norm[feature])
        differences[feature] = {
            "shap": shap_norm[feature],
            "lime": lime_norm[feature],
            "difference": diff,
            "agrees": diff < tolerance,
        }
        if diff < tolerance:
            agreements += 1

    agreement_rate = agreements / len(common_features) if common_features else 0

    # Compare rankings
    shap_rank = sorted(shap_norm.keys(), key=lambda k: shap_norm[k], reverse=True)
    lime_rank = sorted(lime_norm.keys(), key=lambda k: lime_norm[k], reverse=True)

    top_3_match = shap_rank[:3] == lime_rank[:3] if len(common_features) >= 3 else False

    return {
        "comparable": True,
        "agreement_rate": agreement_rate,
        "top_3_features_match": top_3_match,
        "shap_top_3": shap_rank[:3],
        "lime_top_3": lime_rank[:3],
        "feature_differences": differences,
        "common_features": len(common_features),
    }


def compute_lime_hash(weights: Dict[str, float]) -> str:
    """
    Compute SHA-256 hash of LIME weights.

    Args:
        weights: Feature weights dictionary

    Returns:
        64-character hex hash
    """
    sorted_weights = sorted(weights.items())
    return hashlib.sha256(json.dumps(sorted_weights).encode()).hexdigest()
