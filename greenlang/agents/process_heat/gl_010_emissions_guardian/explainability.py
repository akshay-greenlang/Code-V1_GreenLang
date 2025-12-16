# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Explainability Module

This module implements SHAP and LIME explainability for emissions monitoring
models, providing transparent and interpretable AI/ML explanations for
regulatory compliance and stakeholder communication.

Explainability is critical for:
    - Understanding emission prediction model decisions
    - Identifying key drivers of exceedance predictions
    - Regulatory defensibility of AI-assisted monitoring
    - Root cause analysis for emission anomalies
    - Stakeholder communication of complex models

Features:
    - SHAP (SHapley Additive exPlanations) analysis
    - LIME (Local Interpretable Model-agnostic Explanations)
    - Feature importance ranking
    - Counterfactual analysis
    - Explanation caching for performance
    - Visualization-ready outputs

Standards Compliance:
    - EU AI Act transparency requirements
    - TCFD climate disclosure recommendations
    - ISO 14064 verification transparency

Example:
    >>> from greenlang.agents.process_heat.gl_010_emissions_guardian.explainability import (
    ...     SHAPEmissionsAnalyzer,
    ...     LIMEExplainer,
    ...     ExplanationResult,
    ... )
    >>> analyzer = SHAPEmissionsAnalyzer()
    >>> explanation = analyzer.explain_prediction(
    ...     model=emissions_model,
    ...     instance=current_data,
    ...     feature_names=features,
    ... )

Author: GreenLang Process Heat Team
Version: 2.0.0
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
import warnings

import numpy as np
from pydantic import BaseModel, Field, validator

# Conditional imports for optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not installed. Install with: pip install lime")

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class ExplainabilityMethod(str, Enum):
    """Explainability methods available."""
    SHAP_KERNEL = "shap_kernel"
    SHAP_TREE = "shap_tree"
    SHAP_LINEAR = "shap_linear"
    SHAP_DEEP = "shap_deep"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    COUNTERFACTUAL = "counterfactual"


class ExplanationTarget(str, Enum):
    """Target of explanation."""
    PREDICTION = "prediction"
    EXCEEDANCE = "exceedance"
    ANOMALY = "anomaly"
    TREND = "trend"
    COMPLIANCE = "compliance"


class FeatureCategory(str, Enum):
    """Categories for emission-related features."""
    FUEL = "fuel"
    OPERATING_CONDITIONS = "operating_conditions"
    AMBIENT = "ambient"
    EQUIPMENT = "equipment"
    TEMPORAL = "temporal"
    PROCESS = "process"


# =============================================================================
# DATA MODELS
# =============================================================================


class FeatureContribution(BaseModel):
    """
    Individual feature contribution to a prediction.

    Attributes:
        feature_name: Name of the feature
        feature_value: Value of the feature for this instance
        contribution: Contribution to prediction (SHAP value or weight)
        direction: Positive or negative contribution
        rank: Importance rank for this instance
    """

    feature_name: str = Field(
        ...,
        description="Name of the feature"
    )
    feature_value: Any = Field(
        ...,
        description="Value of the feature"
    )
    contribution: float = Field(
        ...,
        description="Contribution to prediction"
    )
    direction: str = Field(
        default="positive",
        description="Direction of contribution (positive/negative)"
    )
    rank: int = Field(
        default=0,
        ge=0,
        description="Importance rank"
    )

    # Additional context
    category: Optional[FeatureCategory] = Field(
        default=None,
        description="Feature category"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Feature unit"
    )
    baseline_value: Optional[float] = Field(
        default=None,
        description="Baseline/average feature value"
    )
    percentile: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Feature value percentile"
    )

    class Config:
        use_enum_values = True


class ExplanationResult(BaseModel):
    """
    Complete explanation result for a prediction.

    Contains SHAP values, feature contributions, and summary
    statistics for model interpretation.

    Attributes:
        explanation_id: Unique explanation identifier
        method: Explainability method used
        target: Target of explanation
        timestamp: Explanation timestamp
    """

    explanation_id: str = Field(
        default_factory=lambda: f"EXP-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hash(datetime.now()) % 10000:04d}",
        description="Unique explanation identifier"
    )
    method: ExplainabilityMethod = Field(
        ...,
        description="Explainability method used"
    )
    target: ExplanationTarget = Field(
        default=ExplanationTarget.PREDICTION,
        description="Target of explanation"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Explanation timestamp"
    )

    # Prediction information
    predicted_value: float = Field(
        ...,
        description="Predicted value being explained"
    )
    predicted_unit: Optional[str] = Field(
        default=None,
        description="Unit of predicted value"
    )
    base_value: Optional[float] = Field(
        default=None,
        description="Base/expected value"
    )

    # Feature contributions
    feature_contributions: List[FeatureContribution] = Field(
        default_factory=list,
        description="Feature contributions to prediction"
    )
    top_positive_features: List[str] = Field(
        default_factory=list,
        description="Top features with positive contribution"
    )
    top_negative_features: List[str] = Field(
        default_factory=list,
        description="Top features with negative contribution"
    )

    # Summary statistics
    total_positive_contribution: float = Field(
        default=0.0,
        description="Sum of positive contributions"
    )
    total_negative_contribution: float = Field(
        default=0.0,
        description="Sum of negative contributions"
    )
    explained_variance: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Fraction of variance explained"
    )

    # Model information
    model_type: Optional[str] = Field(
        default=None,
        description="Type of model explained"
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Model version"
    )

    # Instance information
    instance_id: Optional[str] = Field(
        default=None,
        description="Instance identifier"
    )
    instance_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Instance data snapshot"
    )

    # Confidence
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Confidence in explanation"
    )

    # Human-readable summary
    narrative_summary: str = Field(
        default="",
        description="Human-readable explanation summary"
    )

    # Caching
    cache_key: Optional[str] = Field(
        default=None,
        description="Cache key for this explanation"
    )

    class Config:
        use_enum_values = True

    def get_top_features(self, n: int = 5) -> List[FeatureContribution]:
        """Get top N contributing features."""
        sorted_features = sorted(
            self.feature_contributions,
            key=lambda x: abs(x.contribution),
            reverse=True
        )
        return sorted_features[:n]


class GlobalFeatureImportance(BaseModel):
    """
    Global feature importance for a model.

    Aggregates feature importance across multiple instances
    for overall model interpretation.
    """

    feature_name: str = Field(
        ...,
        description="Feature name"
    )
    importance_score: float = Field(
        ...,
        ge=0,
        description="Global importance score"
    )
    rank: int = Field(
        default=0,
        ge=0,
        description="Importance rank"
    )

    # Statistics
    mean_abs_shap: Optional[float] = Field(
        default=None,
        description="Mean absolute SHAP value"
    )
    std_shap: Optional[float] = Field(
        default=None,
        description="Standard deviation of SHAP values"
    )

    # Direction tendency
    positive_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Percentage of positive contributions"
    )

    # Category
    category: Optional[FeatureCategory] = Field(
        default=None,
        description="Feature category"
    )


class CounterfactualExplanation(BaseModel):
    """
    Counterfactual explanation showing what-if scenarios.

    Shows minimal changes needed to achieve a different outcome.
    """

    explanation_id: str = Field(
        ...,
        description="Explanation identifier"
    )
    original_prediction: float = Field(
        ...,
        description="Original prediction value"
    )
    target_prediction: float = Field(
        ...,
        description="Target prediction value"
    )

    # Changes required
    feature_changes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Feature changes for counterfactual"
    )
    min_changes_required: int = Field(
        default=0,
        description="Minimum features to change"
    )

    # Counterfactual instance
    counterfactual_instance: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Counterfactual feature values"
    )
    counterfactual_prediction: Optional[float] = Field(
        default=None,
        description="Prediction with counterfactual"
    )

    # Feasibility
    feasibility_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Feasibility of changes"
    )

    # Narrative
    narrative: str = Field(
        default="",
        description="Human-readable counterfactual explanation"
    )


# =============================================================================
# SHAP ANALYZER
# =============================================================================


class SHAPEmissionsAnalyzer:
    """
    SHAP-based explainability for emissions models.

    Provides feature attribution using Shapley values for
    transparent model interpretation.

    Features:
        - Kernel SHAP for model-agnostic explanations
        - Tree SHAP for tree-based models
        - Linear SHAP for linear models
        - Feature importance aggregation
        - Explanation caching

    Attributes:
        method: SHAP method to use
        n_samples: Number of samples for SHAP calculation
        cache_enabled: Whether to cache explanations

    Example:
        >>> analyzer = SHAPEmissionsAnalyzer(
        ...     method=ExplainabilityMethod.SHAP_KERNEL,
        ...     n_samples=100,
        ... )
        >>> explanation = analyzer.explain_prediction(
        ...     model=emissions_model,
        ...     instance=feature_values,
        ...     feature_names=["fuel_flow", "o2_pct", "load"],
        ... )
    """

    def __init__(
        self,
        method: ExplainabilityMethod = ExplainabilityMethod.SHAP_KERNEL,
        n_samples: int = 100,
        background_samples: int = 50,
        cache_enabled: bool = True,
        cache_ttl_hours: int = 24,
    ) -> None:
        """
        Initialize SHAP analyzer.

        Args:
            method: SHAP method to use
            n_samples: Number of samples for SHAP calculation
            background_samples: Background samples for kernel SHAP
            cache_enabled: Enable explanation caching
            cache_ttl_hours: Cache TTL in hours
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")

        self.method = method
        self.n_samples = n_samples
        self.background_samples = background_samples
        self.cache_enabled = cache_enabled
        self.cache_ttl_hours = cache_ttl_hours

        self._cache: Dict[str, Tuple[ExplanationResult, datetime]] = {}
        self._explainer = None
        self._background_data = None

        logger.info(f"SHAPEmissionsAnalyzer initialized with {method.value} method")

    def _get_cache_key(
        self,
        instance: np.ndarray,
        model_id: str,
    ) -> str:
        """Generate cache key for an instance."""
        instance_str = json.dumps(instance.tolist(), sort_keys=True)
        combined = f"{model_id}:{instance_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _check_cache(self, cache_key: str) -> Optional[ExplanationResult]:
        """Check cache for existing explanation."""
        if not self.cache_enabled:
            return None

        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if datetime.now(timezone.utc) - timestamp < timedelta(hours=self.cache_ttl_hours):
                logger.debug(f"Cache hit for {cache_key}")
                return result
            else:
                del self._cache[cache_key]

        return None

    def _cache_result(self, cache_key: str, result: ExplanationResult) -> None:
        """Cache explanation result."""
        if self.cache_enabled:
            self._cache[cache_key] = (result, datetime.now(timezone.utc))

    def fit_background(
        self,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Fit background data for SHAP calculations.

        Args:
            background_data: Background dataset for SHAP
            feature_names: Names of features
        """
        # Subsample if needed
        if len(background_data) > self.background_samples:
            indices = np.random.choice(
                len(background_data),
                self.background_samples,
                replace=False
            )
            self._background_data = background_data[indices]
        else:
            self._background_data = background_data

        logger.info(f"Fitted background data with {len(self._background_data)} samples")

    def explain_prediction(
        self,
        model: Any,
        instance: np.ndarray,
        feature_names: List[str],
        feature_categories: Optional[Dict[str, FeatureCategory]] = None,
        feature_units: Optional[Dict[str, str]] = None,
        model_id: str = "emissions_model",
        target: ExplanationTarget = ExplanationTarget.PREDICTION,
    ) -> ExplanationResult:
        """
        Generate SHAP explanation for a single prediction.

        Args:
            model: Trained model with predict method
            instance: Feature values for instance to explain
            feature_names: Names of features
            feature_categories: Category mapping for features
            feature_units: Unit mapping for features
            model_id: Model identifier for caching
            target: Target of explanation

        Returns:
            ExplanationResult with SHAP values

        Example:
            >>> explanation = analyzer.explain_prediction(
            ...     model=rf_model,
            ...     instance=np.array([150.5, 3.2, 85.0]),
            ...     feature_names=["fuel_flow", "o2_pct", "load_pct"],
            ... )
        """
        # Ensure instance is 2D
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        # Check cache
        cache_key = self._get_cache_key(instance, model_id)
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        # Get prediction
        predicted_value = float(model.predict(instance)[0])

        # Create explainer if needed
        if self._explainer is None or self._background_data is None:
            raise ValueError("Background data not fitted. Call fit_background first.")

        # Create explainer based on method
        if self.method == ExplainabilityMethod.SHAP_KERNEL:
            explainer = shap.KernelExplainer(
                model.predict,
                self._background_data,
            )
            shap_values = explainer.shap_values(instance, nsamples=self.n_samples)
        elif self.method == ExplainabilityMethod.SHAP_TREE:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(instance)
        elif self.method == ExplainabilityMethod.SHAP_LINEAR:
            explainer = shap.LinearExplainer(model, self._background_data)
            shap_values = explainer.shap_values(instance)
        else:
            # Default to kernel
            explainer = shap.KernelExplainer(model.predict, self._background_data)
            shap_values = explainer.shap_values(instance, nsamples=self.n_samples)

        # Handle multi-output models
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.array(shap_values).flatten()

        # Get base value
        base_value = float(explainer.expected_value)
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[0])

        # Build feature contributions
        contributions = []
        for i, (name, value, shap_val) in enumerate(zip(
            feature_names,
            instance.flatten(),
            shap_values
        )):
            contribution = FeatureContribution(
                feature_name=name,
                feature_value=float(value),
                contribution=float(shap_val),
                direction="positive" if shap_val > 0 else "negative",
                rank=0,  # Will be set later
                category=feature_categories.get(name) if feature_categories else None,
                unit=feature_units.get(name) if feature_units else None,
            )
            contributions.append(contribution)

        # Sort and rank
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        for i, contrib in enumerate(contributions):
            contrib.rank = i + 1

        # Calculate totals
        total_positive = sum(c.contribution for c in contributions if c.contribution > 0)
        total_negative = sum(c.contribution for c in contributions if c.contribution < 0)

        # Top features
        top_positive = [c.feature_name for c in contributions if c.contribution > 0][:5]
        top_negative = [c.feature_name for c in contributions if c.contribution < 0][:5]

        # Generate narrative
        narrative = self._generate_narrative(
            predicted_value=predicted_value,
            base_value=base_value,
            contributions=contributions,
            target=target,
        )

        result = ExplanationResult(
            method=self.method,
            target=target,
            predicted_value=predicted_value,
            base_value=base_value,
            feature_contributions=contributions,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            total_positive_contribution=total_positive,
            total_negative_contribution=total_negative,
            model_type=type(model).__name__,
            instance_data={name: float(val) for name, val in zip(feature_names, instance.flatten())},
            narrative_summary=narrative,
            cache_key=cache_key,
        )

        # Cache result
        self._cache_result(cache_key, result)

        logger.info(f"Generated SHAP explanation with {len(contributions)} features")

        return result

    def explain_batch(
        self,
        model: Any,
        instances: np.ndarray,
        feature_names: List[str],
        model_id: str = "emissions_model",
    ) -> List[ExplanationResult]:
        """
        Generate explanations for multiple instances.

        Args:
            model: Trained model
            instances: Array of instances (n_samples, n_features)
            feature_names: Feature names
            model_id: Model identifier

        Returns:
            List of ExplanationResult objects
        """
        results = []
        for i, instance in enumerate(instances):
            result = self.explain_prediction(
                model=model,
                instance=instance,
                feature_names=feature_names,
                model_id=model_id,
            )
            result.instance_id = f"instance_{i}"
            results.append(result)

        return results

    def calculate_global_importance(
        self,
        model: Any,
        data: np.ndarray,
        feature_names: List[str],
        n_samples: Optional[int] = None,
    ) -> List[GlobalFeatureImportance]:
        """
        Calculate global feature importance using SHAP.

        Args:
            model: Trained model
            data: Dataset for importance calculation
            feature_names: Feature names
            n_samples: Number of samples to use

        Returns:
            List of GlobalFeatureImportance objects
        """
        # Sample data if needed
        if n_samples and len(data) > n_samples:
            indices = np.random.choice(len(data), n_samples, replace=False)
            data = data[indices]

        # Calculate SHAP values for all instances
        if self._background_data is None:
            self.fit_background(data, feature_names)

        explainer = shap.KernelExplainer(model.predict, self._background_data)
        shap_values = explainer.shap_values(data, nsamples=self.n_samples)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.array(shap_values)

        # Calculate importance metrics
        importance_list = []
        for i, name in enumerate(feature_names):
            feature_shap = shap_values[:, i]
            importance = GlobalFeatureImportance(
                feature_name=name,
                importance_score=float(np.mean(np.abs(feature_shap))),
                mean_abs_shap=float(np.mean(np.abs(feature_shap))),
                std_shap=float(np.std(feature_shap)),
                positive_pct=float(np.mean(feature_shap > 0) * 100),
            )
            importance_list.append(importance)

        # Sort and rank
        importance_list.sort(key=lambda x: x.importance_score, reverse=True)
        for i, imp in enumerate(importance_list):
            imp.rank = i + 1

        logger.info(f"Calculated global importance for {len(feature_names)} features")

        return importance_list

    def _generate_narrative(
        self,
        predicted_value: float,
        base_value: float,
        contributions: List[FeatureContribution],
        target: ExplanationTarget,
    ) -> str:
        """Generate human-readable narrative for explanation."""
        top_3 = contributions[:3]

        narrative = f"The predicted value of {predicted_value:.2f} "
        narrative += f"differs from the baseline ({base_value:.2f}) "

        if top_3:
            narrative += "primarily due to: "
            feature_phrases = []
            for c in top_3:
                direction = "increasing" if c.contribution > 0 else "decreasing"
                feature_phrases.append(
                    f"{c.feature_name}={c.feature_value:.2f} ({direction} by {abs(c.contribution):.2f})"
                )
            narrative += ", ".join(feature_phrases) + "."

        return narrative

    def clear_cache(self) -> int:
        """Clear explanation cache."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} cached explanations")
        return count


# =============================================================================
# LIME EXPLAINER
# =============================================================================


class LIMEExplainer:
    """
    LIME-based explainability for emissions models.

    Provides local interpretable explanations using LIME
    (Local Interpretable Model-agnostic Explanations).

    Features:
        - Tabular data explanation
        - Configurable number of features
        - Discretization support
        - Instance-level interpretability

    Attributes:
        num_features: Number of features in explanation
        num_samples: Samples for LIME perturbation

    Example:
        >>> explainer = LIMEExplainer(num_features=10)
        >>> explanation = explainer.explain_prediction(
        ...     model=emissions_model,
        ...     instance=feature_values,
        ...     training_data=train_data,
        ...     feature_names=features,
        ... )
    """

    def __init__(
        self,
        num_features: int = 10,
        num_samples: int = 5000,
        discretize_continuous: bool = True,
        feature_selection: str = "auto",
    ) -> None:
        """
        Initialize LIME explainer.

        Args:
            num_features: Number of features in explanation
            num_samples: Number of samples for perturbation
            discretize_continuous: Discretize continuous features
            feature_selection: Feature selection method
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not installed. Install with: pip install lime")

        self.num_features = num_features
        self.num_samples = num_samples
        self.discretize_continuous = discretize_continuous
        self.feature_selection = feature_selection

        self._explainer = None

        logger.info(f"LIMEExplainer initialized with {num_features} features")

    def fit(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        categorical_features: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        Fit LIME explainer on training data.

        Args:
            training_data: Training dataset
            feature_names: Names of features
            categorical_features: Indices of categorical features
            class_names: Names of classes (for classification)
        """
        self._explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            categorical_features=categorical_features,
            class_names=class_names,
            discretize_continuous=self.discretize_continuous,
            mode="regression",
        )

        logger.info(f"LIME explainer fitted on {len(training_data)} samples")

    def explain_prediction(
        self,
        model: Any,
        instance: np.ndarray,
        feature_names: List[str],
        training_data: Optional[np.ndarray] = None,
        target: ExplanationTarget = ExplanationTarget.PREDICTION,
    ) -> ExplanationResult:
        """
        Generate LIME explanation for a prediction.

        Args:
            model: Trained model with predict method
            instance: Instance to explain
            feature_names: Feature names
            training_data: Training data (if explainer not fitted)
            target: Target of explanation

        Returns:
            ExplanationResult with LIME weights
        """
        # Ensure 1D instance
        if instance.ndim > 1:
            instance = instance.flatten()

        # Fit explainer if needed
        if self._explainer is None:
            if training_data is None:
                raise ValueError("Training data required for first explanation")
            self.fit(training_data, feature_names)

        # Get prediction
        predicted_value = float(model.predict(instance.reshape(1, -1))[0])

        # Generate explanation
        explanation = self._explainer.explain_instance(
            instance,
            model.predict,
            num_features=self.num_features,
            num_samples=self.num_samples,
        )

        # Extract feature weights
        feature_weights = dict(explanation.as_list())

        # Build contributions
        contributions = []
        for i, (name, value) in enumerate(zip(feature_names, instance)):
            # Find matching weight
            weight = 0.0
            for feat_desc, w in feature_weights.items():
                if name in feat_desc:
                    weight = w
                    break

            contribution = FeatureContribution(
                feature_name=name,
                feature_value=float(value),
                contribution=float(weight),
                direction="positive" if weight > 0 else "negative",
            )
            contributions.append(contribution)

        # Sort and rank
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        for i, contrib in enumerate(contributions):
            contrib.rank = i + 1

        # Calculate totals
        total_positive = sum(c.contribution for c in contributions if c.contribution > 0)
        total_negative = sum(c.contribution for c in contributions if c.contribution < 0)

        # Top features
        top_positive = [c.feature_name for c in contributions if c.contribution > 0][:5]
        top_negative = [c.feature_name for c in contributions if c.contribution < 0][:5]

        # Get intercept (base value)
        base_value = float(explanation.intercept[0]) if hasattr(explanation, 'intercept') else 0.0

        result = ExplanationResult(
            method=ExplainabilityMethod.LIME,
            target=target,
            predicted_value=predicted_value,
            base_value=base_value,
            feature_contributions=contributions,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            total_positive_contribution=total_positive,
            total_negative_contribution=total_negative,
            explained_variance=float(explanation.score) if hasattr(explanation, 'score') else None,
            instance_data={name: float(val) for name, val in zip(feature_names, instance)},
            narrative_summary=self._generate_narrative(contributions),
        )

        logger.info(f"Generated LIME explanation with {len(contributions)} features")

        return result

    def _generate_narrative(self, contributions: List[FeatureContribution]) -> str:
        """Generate narrative for LIME explanation."""
        top_3 = [c for c in contributions if abs(c.contribution) > 0][:3]

        if not top_3:
            return "No significant feature contributions identified."

        narrative = "Key factors: "
        feature_phrases = []
        for c in top_3:
            direction = "increases" if c.contribution > 0 else "decreases"
            feature_phrases.append(f"{c.feature_name} {direction} prediction by {abs(c.contribution):.3f}")

        narrative += ", ".join(feature_phrases) + "."

        return narrative


# =============================================================================
# COUNTERFACTUAL EXPLAINER
# =============================================================================


class CounterfactualExplainer:
    """
    Counterfactual explanation generator.

    Identifies minimal changes needed to achieve a different outcome,
    useful for understanding exceedance thresholds.

    Example:
        >>> explainer = CounterfactualExplainer()
        >>> cf = explainer.find_counterfactual(
        ...     model=emissions_model,
        ...     instance=current_data,
        ...     target_value=25.0,  # Target emission limit
        ... )
    """

    def __init__(
        self,
        step_size: float = 0.1,
        max_iterations: int = 100,
        feature_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """
        Initialize counterfactual explainer.

        Args:
            step_size: Step size for feature changes
            max_iterations: Maximum iterations for search
            feature_constraints: Min/max bounds for features
        """
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.feature_constraints = feature_constraints or {}

        logger.info("CounterfactualExplainer initialized")

    def find_counterfactual(
        self,
        model: Any,
        instance: np.ndarray,
        feature_names: List[str],
        target_value: float,
        target_direction: str = "below",
        mutable_features: Optional[List[str]] = None,
    ) -> CounterfactualExplanation:
        """
        Find counterfactual explanation.

        Args:
            model: Trained model
            instance: Original instance
            feature_names: Feature names
            target_value: Target prediction value
            target_direction: "above" or "below" target
            mutable_features: Features that can be changed

        Returns:
            CounterfactualExplanation
        """
        # Ensure 1D
        if instance.ndim > 1:
            instance = instance.flatten()

        original_pred = float(model.predict(instance.reshape(1, -1))[0])

        # Determine mutable features
        if mutable_features is None:
            mutable_features = feature_names
        mutable_indices = [i for i, name in enumerate(feature_names) if name in mutable_features]

        # Initialize counterfactual
        cf_instance = instance.copy()
        changes = []

        # Simple gradient-free search
        for iteration in range(self.max_iterations):
            current_pred = float(model.predict(cf_instance.reshape(1, -1))[0])

            # Check if target achieved
            if target_direction == "below" and current_pred <= target_value:
                break
            elif target_direction == "above" and current_pred >= target_value:
                break

            # Try modifying each mutable feature
            best_change = None
            best_improvement = 0

            for idx in mutable_indices:
                # Try positive and negative changes
                for direction in [-1, 1]:
                    test_instance = cf_instance.copy()
                    step = self.step_size * direction * abs(instance[idx])
                    test_instance[idx] += step

                    # Apply constraints
                    feature_name = feature_names[idx]
                    if feature_name in self.feature_constraints:
                        min_val, max_val = self.feature_constraints[feature_name]
                        test_instance[idx] = np.clip(test_instance[idx], min_val, max_val)

                    test_pred = float(model.predict(test_instance.reshape(1, -1))[0])

                    if target_direction == "below":
                        improvement = current_pred - test_pred
                    else:
                        improvement = test_pred - current_pred

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_change = (idx, test_instance[idx] - cf_instance[idx])

            if best_change is None:
                break

            # Apply best change
            idx, delta = best_change
            cf_instance[idx] += delta
            changes.append({
                "feature": feature_names[idx],
                "original": float(instance[idx]),
                "counterfactual": float(cf_instance[idx]),
                "change": float(delta),
            })

        cf_pred = float(model.predict(cf_instance.reshape(1, -1))[0])

        # Generate narrative
        if changes:
            change_phrases = [
                f"{c['feature']} from {c['original']:.2f} to {c['counterfactual']:.2f}"
                for c in changes[:3]
            ]
            narrative = f"To achieve prediction {target_direction} {target_value:.2f}, change: " + ", ".join(change_phrases)
        else:
            narrative = "No feasible counterfactual found with current constraints."

        result = CounterfactualExplanation(
            explanation_id=f"CF-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            original_prediction=original_pred,
            target_prediction=target_value,
            feature_changes=changes,
            min_changes_required=len(changes),
            counterfactual_instance={name: float(val) for name, val in zip(feature_names, cf_instance)},
            counterfactual_prediction=cf_pred,
            feasibility_score=1.0 if changes else 0.0,
            narrative=narrative,
        )

        logger.info(f"Found counterfactual with {len(changes)} changes")

        return result


# =============================================================================
# EMISSIONS EXPLAINABILITY MANAGER
# =============================================================================


class EmissionsExplainabilityManager:
    """
    Unified manager for emissions model explainability.

    Provides a single interface for SHAP, LIME, and counterfactual
    explanations for emissions monitoring models.

    Example:
        >>> manager = EmissionsExplainabilityManager()
        >>> manager.configure(
        ...     shap_method=ExplainabilityMethod.SHAP_KERNEL,
        ...     lime_features=10,
        ... )
        >>> explanation = manager.explain(
        ...     model=emissions_model,
        ...     instance=current_data,
        ...     method="shap",
        ... )
    """

    def __init__(self) -> None:
        """Initialize explainability manager."""
        self._shap_analyzer: Optional[SHAPEmissionsAnalyzer] = None
        self._lime_explainer: Optional[LIMEExplainer] = None
        self._cf_explainer: Optional[CounterfactualExplainer] = None

        self._feature_names: List[str] = []
        self._feature_categories: Dict[str, FeatureCategory] = {}
        self._feature_units: Dict[str, str] = {}

        logger.info("EmissionsExplainabilityManager initialized")

    def configure(
        self,
        shap_method: ExplainabilityMethod = ExplainabilityMethod.SHAP_KERNEL,
        shap_samples: int = 100,
        lime_features: int = 10,
        lime_samples: int = 5000,
        feature_names: Optional[List[str]] = None,
        feature_categories: Optional[Dict[str, FeatureCategory]] = None,
        feature_units: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Configure explainability components.

        Args:
            shap_method: SHAP method to use
            shap_samples: Samples for SHAP
            lime_features: Features for LIME
            lime_samples: Samples for LIME
            feature_names: Feature names
            feature_categories: Category mapping
            feature_units: Unit mapping
        """
        if SHAP_AVAILABLE:
            self._shap_analyzer = SHAPEmissionsAnalyzer(
                method=shap_method,
                n_samples=shap_samples,
            )

        if LIME_AVAILABLE:
            self._lime_explainer = LIMEExplainer(
                num_features=lime_features,
                num_samples=lime_samples,
            )

        self._cf_explainer = CounterfactualExplainer()

        if feature_names:
            self._feature_names = feature_names
        if feature_categories:
            self._feature_categories = feature_categories
        if feature_units:
            self._feature_units = feature_units

        logger.info("Explainability components configured")

    def fit_background(self, data: np.ndarray, feature_names: List[str]) -> None:
        """Fit background data for explainers."""
        self._feature_names = feature_names

        if self._shap_analyzer:
            self._shap_analyzer.fit_background(data, feature_names)

        if self._lime_explainer:
            self._lime_explainer.fit(data, feature_names)

        logger.info(f"Fitted background data with {len(data)} samples")

    def explain(
        self,
        model: Any,
        instance: np.ndarray,
        method: str = "shap",
        target: ExplanationTarget = ExplanationTarget.PREDICTION,
    ) -> ExplanationResult:
        """
        Generate explanation using specified method.

        Args:
            model: Trained model
            instance: Instance to explain
            method: "shap" or "lime"
            target: Explanation target

        Returns:
            ExplanationResult
        """
        if method.lower() == "shap":
            if self._shap_analyzer is None:
                raise ValueError("SHAP analyzer not configured")
            return self._shap_analyzer.explain_prediction(
                model=model,
                instance=instance,
                feature_names=self._feature_names,
                feature_categories=self._feature_categories,
                feature_units=self._feature_units,
                target=target,
            )
        elif method.lower() == "lime":
            if self._lime_explainer is None:
                raise ValueError("LIME explainer not configured")
            return self._lime_explainer.explain_prediction(
                model=model,
                instance=instance,
                feature_names=self._feature_names,
                target=target,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def explain_exceedance(
        self,
        model: Any,
        instance: np.ndarray,
        threshold: float,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for exceedance.

        Args:
            model: Trained model
            instance: Instance showing exceedance
            threshold: Exceedance threshold

        Returns:
            Dictionary with multiple explanations
        """
        results = {}

        # SHAP explanation
        if self._shap_analyzer:
            results["shap"] = self.explain(model, instance, "shap", ExplanationTarget.EXCEEDANCE)

        # LIME explanation
        if self._lime_explainer:
            results["lime"] = self.explain(model, instance, "lime", ExplanationTarget.EXCEEDANCE)

        # Counterfactual
        if self._cf_explainer:
            results["counterfactual"] = self._cf_explainer.find_counterfactual(
                model=model,
                instance=instance,
                feature_names=self._feature_names,
                target_value=threshold,
                target_direction="below",
            )

        return results

    def get_global_importance(
        self,
        model: Any,
        data: np.ndarray,
        n_samples: Optional[int] = None,
    ) -> List[GlobalFeatureImportance]:
        """Get global feature importance."""
        if self._shap_analyzer is None:
            raise ValueError("SHAP analyzer not configured")

        return self._shap_analyzer.calculate_global_importance(
            model=model,
            data=data,
            feature_names=self._feature_names,
            n_samples=n_samples,
        )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ExplainabilityMethod",
    "ExplanationTarget",
    "FeatureCategory",
    # Models
    "FeatureContribution",
    "ExplanationResult",
    "GlobalFeatureImportance",
    "CounterfactualExplanation",
    # Analyzers
    "SHAPEmissionsAnalyzer",
    "LIMEExplainer",
    "CounterfactualExplainer",
    "EmissionsExplainabilityManager",
    # Availability flags
    "SHAP_AVAILABLE",
    "LIME_AVAILABLE",
]
