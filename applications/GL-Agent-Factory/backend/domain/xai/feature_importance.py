# -*- coding: utf-8 -*-
"""
Feature Importance Module for GreenLang Agents
==============================================

Provides SHAP-style and LIME-style feature importance calculations
for explaining agent decisions and model outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field


class ImportanceMethod(str, Enum):
    """Methods for calculating feature importance."""
    SHAP = "shap"  # Shapley values
    LIME = "lime"  # Local interpretable model-agnostic
    PERMUTATION = "permutation"  # Permutation importance
    GRADIENT = "gradient"  # Gradient-based
    ATTENTION = "attention"  # Attention weights
    CONTRIBUTION = "contribution"  # Direct contribution


class AttributionType(str, Enum):
    """Types of feature attribution."""
    LOCAL = "local"  # Single prediction explanation
    GLOBAL = "global"  # Model-wide importance
    COHORT = "cohort"  # Group-level importance


@dataclass
class FeatureAttribution:
    """Attribution for a single feature."""
    feature_name: str
    feature_value: Any
    attribution_value: float  # How much this feature contributed
    attribution_percent: float = 0.0  # Percentage of total
    direction: str = "neutral"  # "positive", "negative", "neutral"
    baseline_value: Optional[Any] = None
    unit: str = ""

    @property
    def abs_attribution(self) -> float:
        """Absolute attribution value for ranking."""
        return abs(self.attribution_value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature": self.feature_name,
            "value": self.feature_value,
            "attribution": self.attribution_value,
            "percent": self.attribution_percent,
            "direction": self.direction,
            "unit": self.unit,
        }


class FeatureImportance(BaseModel):
    """
    Complete feature importance analysis result.

    Provides both numerical attribution values and human-readable explanations.
    """
    prediction_value: float
    baseline_value: float

    # Attributions
    attributions: List[Dict[str, Any]] = Field(default_factory=list)

    # Method metadata
    method: ImportanceMethod = ImportanceMethod.CONTRIBUTION
    attribution_type: AttributionType = AttributionType.LOCAL

    # Summary
    top_positive_features: List[str] = Field(default_factory=list)
    top_negative_features: List[str] = Field(default_factory=list)

    # Explanation text
    natural_language_explanation: str = ""

    # Confidence
    confidence_score: float = 1.0

    def get_top_features(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top N features by absolute attribution."""
        sorted_attrs = sorted(
            self.attributions,
            key=lambda x: abs(x.get("attribution", 0)),
            reverse=True
        )
        return sorted_attrs[:n]

    def get_feature_attribution(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get attribution for a specific feature."""
        for attr in self.attributions:
            if attr.get("feature") == feature_name:
                return attr
        return None


class SHAPExplainer:
    """
    Simplified SHAP-style explainer for GreenLang agents.

    Calculates Shapley values for feature attribution using
    a sampling-based approximation.
    """

    def __init__(
        self,
        model_func: Callable[..., float],
        feature_names: List[str],
        baseline_values: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize SHAP explainer.

        Args:
            model_func: Function that takes feature dict and returns prediction
            feature_names: List of feature names
            baseline_values: Baseline/reference values for each feature
        """
        self.model_func = model_func
        self.feature_names = feature_names
        self.baseline_values = baseline_values or {}

    def explain(
        self,
        instance: Dict[str, Any],
        n_samples: int = 100,
        seed: Optional[int] = None,
    ) -> FeatureImportance:
        """
        Calculate SHAP values for a single instance.

        Uses sampling-based Shapley value approximation.

        Args:
            instance: Feature values to explain
            n_samples: Number of sampling iterations
            seed: Random seed for reproducibility

        Returns:
            FeatureImportance with SHAP attributions
        """
        if seed is not None:
            np.random.seed(seed)

        # Calculate prediction and baseline
        prediction = self.model_func(**instance)
        baseline_instance = {
            k: self.baseline_values.get(k, 0)
            for k in self.feature_names
        }
        try:
            baseline = self.model_func(**baseline_instance)
        except Exception:
            baseline = 0.0

        # Calculate Shapley values via sampling
        shapley_values = {f: 0.0 for f in self.feature_names}
        n_features = len(self.feature_names)

        for _ in range(n_samples):
            # Random permutation
            perm = np.random.permutation(self.feature_names).tolist()

            # Calculate marginal contributions
            prev_subset = {}
            for i, feature in enumerate(perm):
                # With feature
                with_feature = prev_subset.copy()
                with_feature[feature] = instance.get(feature, 0)

                # Fill remaining with baseline
                full_with = baseline_instance.copy()
                full_with.update(with_feature)

                full_without = baseline_instance.copy()
                full_without.update(prev_subset)

                try:
                    val_with = self.model_func(**full_with)
                    val_without = self.model_func(**full_without)
                    marginal = val_with - val_without
                except Exception:
                    marginal = 0.0

                shapley_values[feature] += marginal / n_samples
                prev_subset[feature] = instance.get(feature, 0)

        # Build attributions
        total_attribution = sum(abs(v) for v in shapley_values.values())
        attributions = []

        for feature in self.feature_names:
            value = shapley_values[feature]
            pct = (abs(value) / total_attribution * 100) if total_attribution > 0 else 0

            attributions.append({
                "feature": feature,
                "value": instance.get(feature),
                "attribution": round(value, 6),
                "percent": round(pct, 2),
                "direction": "positive" if value > 0 else ("negative" if value < 0 else "neutral"),
                "baseline": self.baseline_values.get(feature),
            })

        # Sort by absolute attribution
        attributions.sort(key=lambda x: abs(x["attribution"]), reverse=True)

        # Get top features
        top_positive = [a["feature"] for a in attributions if a["direction"] == "positive"][:3]
        top_negative = [a["feature"] for a in attributions if a["direction"] == "negative"][:3]

        # Generate explanation
        explanation = self._generate_explanation(attributions, prediction, baseline)

        return FeatureImportance(
            prediction_value=prediction,
            baseline_value=baseline,
            attributions=attributions,
            method=ImportanceMethod.SHAP,
            attribution_type=AttributionType.LOCAL,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            natural_language_explanation=explanation,
        )

    def _generate_explanation(
        self,
        attributions: List[Dict[str, Any]],
        prediction: float,
        baseline: float,
    ) -> str:
        """Generate natural language explanation."""
        parts = [f"Prediction: {prediction:.4f} (baseline: {baseline:.4f})"]
        parts.append("\nKey contributing factors:")

        for attr in attributions[:5]:
            direction = "increased" if attr["direction"] == "positive" else "decreased"
            parts.append(
                f"- {attr['feature']} = {attr['value']} "
                f"{direction} prediction by {abs(attr['attribution']):.4f} "
                f"({attr['percent']:.1f}%)"
            )

        return "\n".join(parts)


class LIMEExplainer:
    """
    LIME-style local explainer for GreenLang agents.

    Fits a local linear model around the instance to explain.
    """

    def __init__(
        self,
        model_func: Callable[..., float],
        feature_names: List[str],
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Initialize LIME explainer.

        Args:
            model_func: Function that takes feature dict and returns prediction
            feature_names: List of feature names
            feature_ranges: Min/max ranges for each feature
        """
        self.model_func = model_func
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges or {}

    def explain(
        self,
        instance: Dict[str, Any],
        n_samples: int = 500,
        kernel_width: float = 0.75,
        seed: Optional[int] = None,
    ) -> FeatureImportance:
        """
        Calculate LIME attributions for a single instance.

        Generates local samples, weights them by distance, and fits
        a weighted linear regression.

        Args:
            instance: Feature values to explain
            n_samples: Number of perturbed samples
            kernel_width: Width of exponential kernel for weighting
            seed: Random seed for reproducibility

        Returns:
            FeatureImportance with LIME attributions
        """
        if seed is not None:
            np.random.seed(seed)

        # Get instance prediction
        prediction = self.model_func(**instance)

        # Generate perturbed samples
        n_features = len(self.feature_names)
        samples = np.zeros((n_samples, n_features))
        predictions = np.zeros(n_samples)
        weights = np.zeros(n_samples)

        instance_array = np.array([instance.get(f, 0) for f in self.feature_names])

        for i in range(n_samples):
            # Perturb features
            perturbed = instance_array.copy()

            for j, feature in enumerate(self.feature_names):
                if feature in self.feature_ranges:
                    low, high = self.feature_ranges[feature]
                    std = (high - low) / 4
                else:
                    std = abs(perturbed[j]) * 0.1 if perturbed[j] != 0 else 1.0

                perturbed[j] += np.random.normal(0, std)

            samples[i] = perturbed

            # Get prediction for perturbed sample
            perturbed_dict = {f: perturbed[k] for k, f in enumerate(self.feature_names)}
            try:
                predictions[i] = self.model_func(**perturbed_dict)
            except Exception:
                predictions[i] = prediction

            # Calculate weight based on distance
            distance = np.sqrt(np.sum((perturbed - instance_array) ** 2))
            weights[i] = np.exp(-(distance ** 2) / (kernel_width ** 2))

        # Fit weighted linear regression
        # Add intercept
        X = np.column_stack([np.ones(n_samples), samples])
        W = np.diag(weights)

        try:
            # Weighted least squares: (X'WX)^-1 X'Wy
            XtW = X.T @ W
            coefficients = np.linalg.solve(XtW @ X, XtW @ predictions)
            feature_coefficients = coefficients[1:]  # Exclude intercept
        except np.linalg.LinAlgError:
            # Fallback to simple average
            feature_coefficients = np.zeros(n_features)

        # Build attributions
        total_attribution = sum(abs(c) for c in feature_coefficients)
        attributions = []

        for j, feature in enumerate(self.feature_names):
            coef = feature_coefficients[j]
            value = instance.get(feature, 0)
            attribution = coef * value  # Linear contribution
            pct = (abs(coef) / total_attribution * 100) if total_attribution > 0 else 0

            attributions.append({
                "feature": feature,
                "value": value,
                "coefficient": round(coef, 6),
                "attribution": round(attribution, 6),
                "percent": round(pct, 2),
                "direction": "positive" if coef > 0 else ("negative" if coef < 0 else "neutral"),
            })

        attributions.sort(key=lambda x: abs(x.get("coefficient", 0)), reverse=True)

        top_positive = [a["feature"] for a in attributions if a["direction"] == "positive"][:3]
        top_negative = [a["feature"] for a in attributions if a["direction"] == "negative"][:3]

        explanation = self._generate_explanation(attributions, prediction)

        return FeatureImportance(
            prediction_value=prediction,
            baseline_value=0.0,  # LIME doesn't use baseline
            attributions=attributions,
            method=ImportanceMethod.LIME,
            attribution_type=AttributionType.LOCAL,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            natural_language_explanation=explanation,
        )

    def _generate_explanation(
        self,
        attributions: List[Dict[str, Any]],
        prediction: float,
    ) -> str:
        """Generate natural language explanation."""
        parts = [f"Prediction: {prediction:.4f}"]
        parts.append("\nLocal linear model coefficients:")

        for attr in attributions[:5]:
            sign = "+" if attr["coefficient"] > 0 else ""
            parts.append(
                f"- {attr['feature']}: {sign}{attr['coefficient']:.4f} "
                f"(value={attr['value']}, impact={attr['attribution']:.4f})"
            )

        return "\n".join(parts)


class PermutationImportance:
    """
    Permutation-based feature importance calculator.

    Measures importance by observing prediction change when
    feature values are shuffled.
    """

    def __init__(
        self,
        model_func: Callable[..., float],
        feature_names: List[str],
    ):
        self.model_func = model_func
        self.feature_names = feature_names

    def calculate_importance(
        self,
        dataset: List[Dict[str, Any]],
        n_repeats: int = 10,
        seed: Optional[int] = None,
    ) -> FeatureImportance:
        """
        Calculate global permutation importance.

        Args:
            dataset: List of instances to evaluate
            n_repeats: Number of shuffling iterations
            seed: Random seed

        Returns:
            FeatureImportance with global attributions
        """
        if seed is not None:
            np.random.seed(seed)

        n_samples = len(dataset)

        # Calculate baseline predictions
        baseline_preds = np.array([self.model_func(**x) for x in dataset])
        baseline_metric = np.mean(baseline_preds ** 2)  # MSE-like

        importances = {}

        for feature in self.feature_names:
            feature_importances = []

            for _ in range(n_repeats):
                # Shuffle feature values
                shuffled_values = [d[feature] for d in dataset]
                np.random.shuffle(shuffled_values)

                # Create shuffled dataset
                shuffled_data = []
                for i, instance in enumerate(dataset):
                    shuffled_instance = instance.copy()
                    shuffled_instance[feature] = shuffled_values[i]
                    shuffled_data.append(shuffled_instance)

                # Calculate predictions with shuffled feature
                shuffled_preds = np.array([self.model_func(**x) for x in shuffled_data])
                shuffled_metric = np.mean(shuffled_preds ** 2)

                # Importance = degradation from shuffling
                importance = shuffled_metric - baseline_metric
                feature_importances.append(importance)

            importances[feature] = np.mean(feature_importances)

        # Build attributions
        total = sum(abs(v) for v in importances.values())
        attributions = []

        for feature in self.feature_names:
            imp = importances[feature]
            pct = (abs(imp) / total * 100) if total > 0 else 0
            attributions.append({
                "feature": feature,
                "importance": round(imp, 6),
                "percent": round(pct, 2),
            })

        attributions.sort(key=lambda x: abs(x["importance"]), reverse=True)

        return FeatureImportance(
            prediction_value=np.mean(baseline_preds),
            baseline_value=baseline_metric,
            attributions=attributions,
            method=ImportanceMethod.PERMUTATION,
            attribution_type=AttributionType.GLOBAL,
            top_positive_features=[a["feature"] for a in attributions[:3]],
        )


def calculate_feature_importance(
    model_func: Callable[..., float],
    instance: Dict[str, Any],
    feature_names: List[str],
    method: ImportanceMethod = ImportanceMethod.CONTRIBUTION,
    baseline_values: Optional[Dict[str, Any]] = None,
) -> FeatureImportance:
    """
    High-level function to calculate feature importance.

    Args:
        model_func: Prediction function
        instance: Instance to explain
        feature_names: List of feature names
        method: Attribution method to use
        baseline_values: Baseline values for comparison

    Returns:
        FeatureImportance result
    """
    if method == ImportanceMethod.SHAP:
        explainer = SHAPExplainer(model_func, feature_names, baseline_values)
        return explainer.explain(instance)

    elif method == ImportanceMethod.LIME:
        explainer = LIMEExplainer(model_func, feature_names)
        return explainer.explain(instance)

    else:  # CONTRIBUTION - simple direct contribution
        prediction = model_func(**instance)
        baseline = model_func(**{f: 0 for f in feature_names}) if baseline_values is None else model_func(**baseline_values)

        # Calculate marginal contributions
        attributions = []
        total_contrib = 0.0

        for feature in feature_names:
            # Remove feature and see impact
            reduced = instance.copy()
            reduced[feature] = baseline_values.get(feature, 0) if baseline_values else 0

            try:
                reduced_pred = model_func(**reduced)
                contribution = prediction - reduced_pred
            except Exception:
                contribution = 0.0

            total_contrib += abs(contribution)

            attributions.append({
                "feature": feature,
                "value": instance.get(feature),
                "attribution": round(contribution, 6),
                "direction": "positive" if contribution > 0 else ("negative" if contribution < 0 else "neutral"),
            })

        # Add percentages
        for attr in attributions:
            attr["percent"] = round(
                (abs(attr["attribution"]) / total_contrib * 100) if total_contrib > 0 else 0,
                2
            )

        attributions.sort(key=lambda x: abs(x["attribution"]), reverse=True)

        return FeatureImportance(
            prediction_value=prediction,
            baseline_value=baseline,
            attributions=attributions,
            method=method,
            attribution_type=AttributionType.LOCAL,
            top_positive_features=[a["feature"] for a in attributions if a["direction"] == "positive"][:3],
            top_negative_features=[a["feature"] for a in attributions if a["direction"] == "negative"][:3],
        )
