# -*- coding: utf-8 -*-
"""
LIME Explainer Module

This module provides LIME (Local Interpretable Model-agnostic Explanations)
integration for GreenLang ML models, enabling local explanation of individual
predictions with interpretable surrogate models.

LIME explains predictions by approximating the model locally with an
interpretable model (e.g., linear regression) around the instance being
explained, providing human-understandable feature contributions.

Example:
    >>> from greenlang.ml.explainability import LIMEExplainer
    >>> explainer = LIMEExplainer(model, feature_names=["fuel", "qty", "region"])
    >>> result = explainer.explain_instance(X_sample[0])
    >>> print(result.local_explanation)
"""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class LIMEMode(str, Enum):
    """LIME explanation modes."""
    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"


class KernelType(str, Enum):
    """Kernel types for LIME."""
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    LINEAR = "linear"


class LIMEExplainerConfig(BaseModel):
    """Configuration for LIME explainer."""

    mode: LIMEMode = Field(
        default=LIMEMode.TABULAR,
        description="LIME explanation mode"
    )
    kernel_width: float = Field(
        default=0.75,
        gt=0,
        description="Kernel width for locality"
    )
    kernel_type: KernelType = Field(
        default=KernelType.EXPONENTIAL,
        description="Kernel function type"
    )
    feature_names: Optional[List[str]] = Field(
        default=None,
        description="Names of features"
    )
    categorical_features: Optional[List[int]] = Field(
        default=None,
        description="Indices of categorical features"
    )
    class_names: Optional[List[str]] = Field(
        default=None,
        description="Names of classes for classification"
    )
    num_samples: int = Field(
        default=5000,
        ge=100,
        le=50000,
        description="Number of perturbation samples"
    )
    num_features: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of features to include in explanation"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    discretize_continuous: bool = Field(
        default=True,
        description="Discretize continuous features"
    )
    discretizer: str = Field(
        default="quartile",
        description="Discretization method (quartile, decile, entropy)"
    )
    random_state: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    @validator("discretizer")
    def validate_discretizer(cls, v):
        """Validate discretizer method."""
        valid = ["quartile", "decile", "entropy"]
        if v not in valid:
            raise ValueError(f"Discretizer must be one of {valid}")
        return v


class LIMEResult(BaseModel):
    """Result from LIME explanation."""

    local_explanation: Dict[str, float] = Field(
        ...,
        description="Feature contributions for this instance"
    )
    intercept: float = Field(
        ...,
        description="Intercept of local model"
    )
    local_prediction: float = Field(
        ...,
        description="Prediction from local surrogate model"
    )
    model_prediction: float = Field(
        ...,
        description="Actual model prediction"
    )
    r_squared: float = Field(
        ...,
        description="R-squared of local model fit"
    )
    feature_weights: List[Tuple[str, float]] = Field(
        ...,
        description="Sorted feature weights"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing duration"
    )
    num_samples_used: int = Field(
        ...,
        description="Number of perturbation samples used"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of explanation"
    )


class LIMEBatchResult(BaseModel):
    """Result from batch LIME explanation."""

    explanations: List[LIMEResult] = Field(
        ...,
        description="Individual LIME explanations"
    )
    aggregate_importance: Dict[str, float] = Field(
        ...,
        description="Aggregated feature importance across instances"
    )
    provenance_hash: str = Field(
        ...,
        description="Batch provenance hash"
    )
    processing_time_ms: float = Field(
        ...,
        description="Total processing time"
    )


class LIMEExplainer:
    """
    LIME Explainer for GreenLang ML models.

    This class provides LIME-based local explanations for ML model predictions,
    creating interpretable surrogate models that approximate the behavior of
    complex models in the neighborhood of specific instances.

    LIME works by:
    1. Generating perturbed samples around the instance
    2. Getting model predictions for perturbed samples
    3. Weighting samples by proximity to original instance
    4. Training interpretable model on weighted samples
    5. Extracting feature weights as explanations

    Attributes:
        model: The ML model to explain
        config: Configuration for the explainer
        _explainer: Internal LIME explainer instance
        _training_data_stats: Statistics from training data

    Example:
        >>> model = train_emission_classifier(X_train, y_train)
        >>> explainer = LIMEExplainer(
        ...     model,
        ...     config=LIMEExplainerConfig(
        ...         feature_names=["fuel", "quantity", "region"],
        ...         class_names=["low", "medium", "high"]
        ...     ),
        ...     training_data=X_train
        ... )
        >>> result = explainer.explain_instance(X_test[0])
        >>> print(f"Top factor: {result.feature_weights[0]}")
    """

    def __init__(
        self,
        model: Any,
        config: Optional[LIMEExplainerConfig] = None,
        training_data: Optional[np.ndarray] = None
    ):
        """
        Initialize LIME explainer.

        Args:
            model: ML model with predict/predict_proba method
            config: Explainer configuration
            training_data: Training data for statistics (recommended)
        """
        self.model = model
        self.config = config or LIMEExplainerConfig()
        self._training_data = training_data
        self._explainer = None
        self._initialized = False
        self._training_data_stats: Dict[str, Any] = {}

        # Compute training data statistics if provided
        if training_data is not None:
            self._compute_training_stats(training_data)

        logger.info(
            f"LIMEExplainer initialized with mode={self.config.mode}"
        )

    def _compute_training_stats(self, training_data: np.ndarray) -> None:
        """
        Compute statistics from training data.

        Args:
            training_data: Training dataset
        """
        self._training_data_stats = {
            "mean": np.mean(training_data, axis=0).tolist(),
            "std": np.std(training_data, axis=0).tolist(),
            "min": np.min(training_data, axis=0).tolist(),
            "max": np.max(training_data, axis=0).tolist(),
            "n_features": training_data.shape[1],
            "n_samples": training_data.shape[0]
        }

    def _get_prediction_function(self) -> Callable:
        """
        Get the prediction function from the model.

        Returns:
            Callable prediction function
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba
        elif hasattr(self.model, "predict"):
            # Wrap predict for classification
            def predict_fn(X):
                predictions = self.model.predict(X)
                if len(predictions.shape) == 1:
                    # Binary or regression - return as 2D
                    return np.column_stack([1 - predictions, predictions])
                return predictions
            return predict_fn
        else:
            raise ValueError(
                "Model must have 'predict' or 'predict_proba' method"
            )

    def _initialize_explainer(self, sample: np.ndarray) -> None:
        """
        Initialize the LIME explainer.

        Args:
            sample: Sample data for initialization
        """
        try:
            from lime import lime_tabular
        except ImportError:
            raise ImportError(
                "LIME is required. Install with: pip install lime"
            )

        if self.config.mode != LIMEMode.TABULAR:
            raise NotImplementedError(
                f"Mode {self.config.mode} not yet implemented"
            )

        # Determine feature names
        n_features = sample.shape[0] if len(sample.shape) == 1 else sample.shape[1]

        if self.config.feature_names is not None:
            feature_names = self.config.feature_names
        else:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Use training data if available, otherwise use sample
        if self._training_data is not None:
            training_data = self._training_data
        else:
            # Create synthetic training data around sample
            logger.warning(
                "No training data provided, using synthetic perturbations"
            )
            if len(sample.shape) == 1:
                sample = sample.reshape(1, -1)
            training_data = self._generate_synthetic_training(sample)

        self._explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=self.config.class_names,
            categorical_features=self.config.categorical_features,
            discretize_continuous=self.config.discretize_continuous,
            discretizer=self.config.discretizer,
            kernel_width=self.config.kernel_width,
            random_state=self.config.random_state
        )

        self._initialized = True
        logger.info("LIME explainer initialized successfully")

    def _generate_synthetic_training(
        self,
        sample: np.ndarray,
        n_synthetic: int = 1000
    ) -> np.ndarray:
        """
        Generate synthetic training data around sample.

        Args:
            sample: Reference sample
            n_synthetic: Number of synthetic samples

        Returns:
            Synthetic training data
        """
        np.random.seed(self.config.random_state)

        n_features = sample.shape[1]

        # Generate perturbations around sample
        noise = np.random.normal(0, 0.1, size=(n_synthetic, n_features))
        synthetic = sample + noise * np.abs(sample + 1e-10)

        return synthetic

    def _calculate_provenance(
        self,
        instance: np.ndarray,
        explanation: Dict[str, float]
    ) -> str:
        """
        Calculate SHA-256 provenance hash.

        Args:
            instance: Input instance
            explanation: LIME explanation

        Returns:
            SHA-256 hash string
        """
        input_str = np.array2string(instance, precision=8, separator=",")
        explanation_str = str(sorted(explanation.items()))
        combined = f"{input_str}|{explanation_str}|lime"

        return hashlib.sha256(combined.encode()).hexdigest()

    def explain_instance(
        self,
        instance: Union[np.ndarray, List[float]],
        labels: Optional[List[int]] = None
    ) -> LIMEResult:
        """
        Generate LIME explanation for a single instance.

        This method creates a local interpretable explanation by:
        1. Generating perturbations around the instance
        2. Weighting samples by proximity
        3. Training a local linear model
        4. Returning feature weights as explanation

        Args:
            instance: Single instance to explain
            labels: Class labels to explain (for classification)

        Returns:
            LIMEResult with local explanation

        Raises:
            ValueError: If instance is invalid

        Example:
            >>> result = explainer.explain_instance(X_test[0])
            >>> for feature, weight in result.feature_weights[:5]:
            ...     print(f"{feature}: {weight:+.4f}")
        """
        start_time = datetime.utcnow()

        # Convert to numpy array
        if isinstance(instance, list):
            instance = np.array(instance)

        if len(instance.shape) > 1:
            instance = instance.flatten()

        # Initialize explainer if needed
        if not self._initialized:
            self._initialize_explainer(instance)

        # Get model prediction
        predict_fn = self._get_prediction_function()
        model_pred = predict_fn(instance.reshape(1, -1))[0]

        if isinstance(model_pred, np.ndarray):
            model_prediction = float(model_pred[1]) if len(model_pred) > 1 else float(model_pred[0])
        else:
            model_prediction = float(model_pred)

        # Generate LIME explanation
        logger.info("Generating LIME explanation")

        if labels is None:
            labels = [1] if self.config.class_names else [0]

        explanation = self._explainer.explain_instance(
            instance,
            predict_fn,
            num_features=self.config.num_features,
            num_samples=self.config.num_samples,
            labels=labels
        )

        # Extract explanation details
        label = labels[0]
        exp_list = explanation.as_list(label=label)

        # Parse feature weights
        feature_weights = []
        local_explanation = {}

        for feature_desc, weight in exp_list:
            # Extract feature name from description
            feature_name = feature_desc.split()[0].split("<")[0].split(">")[0]
            feature_weights.append((feature_desc, float(weight)))
            local_explanation[feature_desc] = float(weight)

        # Get local model details
        local_pred = explanation.local_pred[label] if hasattr(explanation, "local_pred") else model_prediction

        intercept = explanation.intercept[label] if hasattr(explanation, "intercept") else 0.0

        # Calculate R-squared
        r_squared = explanation.score if hasattr(explanation, "score") else 0.0

        # Calculate provenance
        provenance_hash = self._calculate_provenance(instance, local_explanation)

        # Calculate processing time
        processing_time_ms = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000

        logger.info(
            f"LIME explanation completed in {processing_time_ms:.2f}ms, "
            f"R-squared: {r_squared:.4f}"
        )

        return LIMEResult(
            local_explanation=local_explanation,
            intercept=float(intercept),
            local_prediction=float(local_pred),
            model_prediction=model_prediction,
            r_squared=float(r_squared),
            feature_weights=feature_weights,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
            num_samples_used=self.config.num_samples,
            timestamp=datetime.utcnow()
        )

    def explain_batch(
        self,
        instances: Union[np.ndarray, List[List[float]]],
        labels: Optional[List[int]] = None
    ) -> LIMEBatchResult:
        """
        Generate LIME explanations for multiple instances.

        Args:
            instances: Multiple instances to explain
            labels: Class labels to explain

        Returns:
            LIMEBatchResult with individual and aggregate explanations

        Example:
            >>> batch_result = explainer.explain_batch(X_test[:10])
            >>> print(f"Aggregate importance: {batch_result.aggregate_importance}")
        """
        start_time = datetime.utcnow()

        if isinstance(instances, list):
            instances = np.array(instances)

        explanations = []
        importance_accumulator: Dict[str, List[float]] = {}

        for i, instance in enumerate(instances):
            logger.info(f"Explaining instance {i+1}/{len(instances)}")
            result = self.explain_instance(instance, labels)
            explanations.append(result)

            # Accumulate importance
            for feature, weight in result.local_explanation.items():
                if feature not in importance_accumulator:
                    importance_accumulator[feature] = []
                importance_accumulator[feature].append(abs(weight))

        # Calculate aggregate importance
        aggregate_importance = {
            feature: float(np.mean(weights))
            for feature, weights in importance_accumulator.items()
        }

        # Sort by importance
        aggregate_importance = dict(
            sorted(
                aggregate_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )

        # Calculate batch provenance
        batch_data = str([e.provenance_hash for e in explanations])
        provenance_hash = hashlib.sha256(batch_data.encode()).hexdigest()

        processing_time_ms = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000

        return LIMEBatchResult(
            explanations=explanations,
            aggregate_importance=aggregate_importance,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms
        )

    def get_feature_stability(
        self,
        instance: np.ndarray,
        n_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Assess stability of LIME explanations through multiple runs.

        Args:
            instance: Instance to explain
            n_runs: Number of explanation runs

        Returns:
            Dictionary with stability metrics
        """
        explanations = []

        for i in range(n_runs):
            result = self.explain_instance(instance)
            explanations.append(result.local_explanation)

        # Calculate consistency
        all_features = set()
        for exp in explanations:
            all_features.update(exp.keys())

        feature_stability = {}
        for feature in all_features:
            weights = [
                exp.get(feature, 0.0) for exp in explanations
            ]
            feature_stability[feature] = {
                "mean": float(np.mean(weights)),
                "std": float(np.std(weights)),
                "cv": float(np.std(weights) / (np.mean(weights) + 1e-10))
            }

        return {
            "n_runs": n_runs,
            "feature_stability": feature_stability,
            "mean_cv": float(np.mean([
                s["cv"] for s in feature_stability.values()
            ]))
        }


# Unit test stubs



    def generate_report(self, explanations, output_format="html"):
        """Generate human-readable report from explanations."""
        if output_format == "json":
            return self._generate_json_report(explanations)
        elif output_format == "markdown":
            return self._generate_markdown_report(explanations)
        else:
            return self._generate_html_report(explanations)

    def _generate_html_report(self, explanation):
        """Generate HTML report from explanation."""
        if isinstance(explanation, LIMEBatchResult):
            return self._generate_batch_html_report(explanation)

        html = f"""<html><head><style>
body {{font-family: Arial, sans-serif; margin: 20px;}}
.metric {{background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px;}}
table {{border-collapse: collapse; width: 100%;}}
th, td {{border: 1px solid #ddd; padding: 8px; text-align: left;}}
th {{background-color: #007bff; color: white;}}
</style></head><body><h1>LIME Explanation Report</h1>
<div class="metric"><strong>Model Prediction:</strong> {explanation.model_prediction:.4f}</div>
<div class="metric"><strong>Local Prediction:</strong> {explanation.local_prediction:.4f}</div>
<div class="metric"><strong>R-squared:</strong> {explanation.r_squared:.4f}</div>
<h2>Feature Contributions</h2><table><tr><th>Feature</th><th>Weight</th></tr>"""

        for feature, weight in explanation.feature_weights[:20]:
            html += f'<tr><td>{feature}</td><td>{weight:+.4f}</td></tr>'

        html += f"""</table><p style="color: #666;">Provenance: <code>{explanation.provenance_hash[:16]}...</code></p>
</body></html>"""
        return html

    def _generate_batch_html_report(self, batch_result):
        """Generate HTML report for batch explanations."""
        html = f"""<html><head><style>
body {{font-family: Arial, sans-serif; margin: 20px;}}
table {{border-collapse: collapse; width: 100%;}}
th, td {{border: 1px solid #ddd; padding: 8px;}}
th {{background-color: #007bff; color: white;}}
</style></head><body><h1>Batch LIME Report</h1>
<p><strong>Instances:</strong> {len(batch_result.explanations)}</p>
<h2>Feature Importance</h2><table><tr><th>Feature</th><th>Importance</th></tr>"""

        for feature, importance in sorted(batch_result.aggregate_importance.items(), key=lambda x: x[1], reverse=True)[:20]:
            html += f"<tr><td>{feature}</td><td>{importance:.4f}</td></tr>"

        html += "</table></body></html>"
        return html

    def _generate_json_report(self, explanation):
        """Generate JSON report from explanation."""
        import json
        if isinstance(explanation, LIMEBatchResult):
            data = {
                "type": "batch",
                "num_instances": len(explanation.explanations),
                "aggregate_importance": explanation.aggregate_importance,
                "provenance_hash": explanation.provenance_hash
            }
        else:
            data = {
                "type": "single",
                "model_prediction": explanation.model_prediction,
                "local_prediction": explanation.local_prediction,
                "r_squared": explanation.r_squared,
                "feature_weights": dict(explanation.feature_weights),
                "provenance_hash": explanation.provenance_hash
            }
        return json.dumps(data, indent=2)

    def _generate_markdown_report(self, explanation):
        """Generate Markdown report from explanation."""
        if isinstance(explanation, LIMEBatchResult):
            md = f"""# Batch LIME Explanations

- Instances: {len(explanation.explanations)}

## Feature Importance

| Feature | Importance |
|---------|-----------|
"""
            for feature, importance in sorted(explanation.aggregate_importance.items(), key=lambda x: x[1], reverse=True)[:20]:
                md += f"| {feature} | {importance:.4f} |\n"
        else:
            md = f"""# LIME Explanation Report

- Model Prediction: {explanation.model_prediction:.4f}
- Local Prediction: {explanation.local_prediction:.4f}
- R-squared: {explanation.r_squared:.4f}

## Feature Contributions

| Feature | Weight |
|---------|--------|
"""
            for feature, weight in explanation.feature_weights[:20]:
                md += f"| {feature} | {weight:+.4f} |\n"

        return md


class ProcessHeatLIMEExplainer(LIMEExplainer):
    """Process Heat LIME Explainer with caching support."""

    def __init__(self, model, config=None, training_data=None, cache_size=1000):
        super().__init__(model, config, training_data)
        self._explanation_cache = {}
        self.cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info(f"ProcessHeatLIMEExplainer initialized with cache_size={cache_size}")

    def _get_cache_key(self, instance):
        return hashlib.sha256(
            np.array2string(instance, precision=8, separator=",").encode()
        ).hexdigest()

    def explain_instance(self, instance, labels=None, use_cache=True):
        if isinstance(instance, list):
            instance = np.array(instance)

        cache_key = self._get_cache_key(instance)

        if use_cache and cache_key in self._explanation_cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit (hits={self._cache_hits}, misses={self._cache_misses})")
            return self._explanation_cache[cache_key]

        self._cache_misses += 1
        result = super().explain_instance(instance, labels)

        if len(self._explanation_cache) < self.cache_size:
            self._explanation_cache[cache_key] = result

        return result

    def get_cache_stats(self):
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total if total > 0 else 0)
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cached_items": len(self._explanation_cache),
            "cache_size": self.cache_size
        }

    def clear_cache(self):
        self._explanation_cache.clear()
        logger.info("Explanation cache cleared")


class GL001LIMEExplainer(ProcessHeatLIMEExplainer):
    """LIME Explainer for GL001 Thermal Command Orchestrator."""

    def __init__(self, model, config=None, training_data=None):
        if config is None:
            config = LIMEExplainerConfig(
                feature_names=[
                    "setpoint_temp", "current_temp", "boiler_power",
                    "demand_forecast", "weather_temp", "system_efficiency",
                    "fuel_cost", "grid_price", "thermal_load", "ambient_humidity"
                ],
                class_names=["low_power", "medium_power", "high_power"],
                num_features=8
            )
        super().__init__(model, config, training_data)
        logger.info("GL001LIMEExplainer initialized for orchestrator decisions")

    def explain_decision(self, instance, decision_type="power_level"):
        result = self.explain_instance(instance)
        return {
            "decision_type": decision_type,
            "model_prediction": result.model_prediction,
            "explanation": result.local_explanation,
            "top_factors": result.feature_weights[:5],
            "confidence": result.r_squared,
            "provenance_hash": result.provenance_hash
        }


class GL010LIMEExplainer(ProcessHeatLIMEExplainer):
    """LIME Explainer for GL010 Emissions Guardian."""

    def __init__(self, model, config=None, training_data=None):
        if config is None:
            config = LIMEExplainerConfig(
                feature_names=[
                    "fuel_type", "fuel_quantity", "combustion_efficiency",
                    "emission_factor", "co2_content", "ch4_content",
                    "n2o_content", "operating_hours", "temperature", "oxygen_level"
                ],
                class_names=["low_emissions", "medium_emissions", "high_emissions"],
                num_features=8
            )
        super().__init__(model, config, training_data)
        logger.info("GL010LIMEExplainer initialized for emissions predictions")

    def explain_emission_prediction(self, instance, emission_scope="scope1"):
        result = self.explain_instance(instance)
        return {
            "emission_scope": emission_scope,
            "predicted_emissions": result.model_prediction,
            "local_model_emissions": result.local_prediction,
            "contributing_factors": result.local_explanation,
            "top_contributors": result.feature_weights[:5],
            "model_reliability": result.r_squared,
            "provenance_hash": result.provenance_hash
        }


class GL013LIMEExplainer(ProcessHeatLIMEExplainer):
    """LIME Explainer for GL013 Predictive Maintenance."""

    def __init__(self, model, config=None, training_data=None):
        if config is None:
            config = LIMEExplainerConfig(
                feature_names=[
                    "equipment_age", "operating_hours", "vibration_level",
                    "temperature_trend", "pressure_diff", "motor_current",
                    "efficiency_decline", "maintenance_history", "failure_rate",
                    "component_condition"
                ],
                class_names=["healthy", "warning", "failure_imminent"],
                num_features=8
            )
        super().__init__(model, config, training_data)
        logger.info("GL013LIMEExplainer initialized for failure predictions")

    def explain_failure_prediction(self, instance, equipment_id=""):
        result = self.explain_instance(instance)
        failure_probability = result.model_prediction
        return {
            "equipment_id": equipment_id,
            "failure_probability": failure_probability,
            "failure_risk_level": self._classify_risk(failure_probability),
            "contributing_factors": result.local_explanation,
            "top_risk_factors": result.feature_weights[:5],
            "model_confidence": result.r_squared,
            "provenance_hash": result.provenance_hash
        }

    @staticmethod
    def _classify_risk(probability):
        if probability < 0.3:
            return "low"
        elif probability < 0.7:
            return "medium"
        else:
            return "high"
