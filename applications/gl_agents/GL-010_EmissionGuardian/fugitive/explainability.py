# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Fugitive Detection ML Explainability

This module provides SHAP and LIME-based explanations for the fugitive emissions
anomaly detection models. Ensures human reviewers can understand WHY an anomaly
was flagged.

Zero-Hallucination Principle:
    - Explanations are based on actual model internals, not generated text
    - All feature contributions are mathematically derived
    - Provenance tracking for audit trails

Supported Methods:
    - SHAP TreeExplainer for Isolation Forest
    - SHAP KernelExplainer for any model
    - LIME for local interpretable explanations
    - Statistical feature importance

Author: GreenLang GL-010 EmissionsGuardian
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import json
import uuid

import numpy as np

logger = logging.getLogger(__name__)


class ExplainabilityMethod(str, Enum):
    """Available explainability methods."""
    SHAP_TREE = "shap_tree"
    SHAP_KERNEL = "shap_kernel"
    LIME = "lime"
    PERMUTATION = "permutation"
    STATISTICAL = "statistical"


class ContributionDirection(str, Enum):
    """Direction of feature contribution."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class FeatureContribution:
    """Individual feature contribution to prediction."""
    feature_name: str
    feature_value: float
    contribution_value: float
    contribution_direction: ContributionDirection
    contribution_percent: float
    rank: int
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature_name,
            "value": round(self.feature_value, 4),
            "contribution": round(self.contribution_value, 4),
            "direction": self.contribution_direction.value,
            "percent": round(self.contribution_percent, 2),
            "rank": self.rank,
        }


@dataclass
class ModelExplanation:
    """Complete explanation for an ML prediction."""
    explanation_id: str
    prediction_id: str
    model_id: str
    model_version: str
    timestamp: datetime

    # Prediction details
    prediction_value: float
    prediction_label: str
    confidence: float

    # Method used
    method: ExplainabilityMethod
    baseline_value: float

    # Feature contributions (sorted by importance)
    feature_contributions: List[FeatureContribution]

    # Raw SHAP/LIME values
    shap_values: Optional[Dict[str, float]] = None
    lime_weights: Optional[Dict[str, float]] = None

    # Summary for human review
    summary_text: str = ""
    detailed_explanation: str = ""

    # Provenance
    provenance_hash: str = ""
    input_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "explanation_id": self.explanation_id,
            "prediction_id": self.prediction_id,
            "model_id": self.model_id,
            "method": self.method.value,
            "prediction_value": round(self.prediction_value, 4),
            "confidence": round(self.confidence, 4),
            "top_factors": [c.to_dict() for c in self.feature_contributions[:5]],
            "summary": self.summary_text,
            "provenance_hash": self.provenance_hash,
        }


class SHAPExplainer:
    """
    SHAP-based explainer for anomaly detection models.

    Provides TreeExplainer for Isolation Forest and KernelExplainer
    for general models. Ensures all explanations are mathematically
    grounded, not hallucinated.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_id: str = "unknown",
        model_version: str = "1.0.0",
    ):
        self.model = model
        self.feature_names = feature_names
        self.model_id = model_id
        self.model_version = model_version
        self._explainer = None
        self._baseline = None
        self._is_initialized = False

    def initialize(self, background_data: np.ndarray) -> bool:
        """
        Initialize SHAP explainer with background data.

        Args:
            background_data: Sample of training data for baseline

        Returns:
            True if initialization successful
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not available - using permutation fallback")
            return False

        try:
            # Try TreeExplainer first (faster for tree models)
            if hasattr(self.model, 'estimators_'):
                self._explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized SHAP TreeExplainer")
            else:
                # Fall back to KernelExplainer
                self._explainer = shap.KernelExplainer(
                    self.model.predict if hasattr(self.model, 'predict') else self.model,
                    background_data
                )
                logger.info("Initialized SHAP KernelExplainer")

            self._baseline = np.mean(background_data, axis=0)
            self._is_initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize SHAP: {e}")
            return False

    def explain(
        self,
        feature_vector: np.ndarray,
        prediction_value: float,
        prediction_id: str,
    ) -> ModelExplanation:
        """
        Generate SHAP explanation for a prediction.

        Args:
            feature_vector: Input features (1D array)
            prediction_value: Model prediction
            prediction_id: Unique prediction identifier

        Returns:
            ModelExplanation with SHAP values
        """
        explanation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        if not self._is_initialized or self._explainer is None:
            # Fallback to permutation importance
            return self._permutation_explain(
                feature_vector, prediction_value, prediction_id
            )

        try:
            import shap

            # Reshape for single prediction
            X = feature_vector.reshape(1, -1)

            # Get SHAP values
            shap_values = self._explainer.shap_values(X)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # For multi-output models, take first output
                sv = shap_values[0][0] if len(shap_values) > 0 else shap_values[0]
            else:
                sv = shap_values[0]

            # Build feature contributions
            shap_dict = {
                name: float(sv[i])
                for i, name in enumerate(self.feature_names)
            }

            contributions = self._build_contributions(shap_dict, feature_vector)

            # Calculate baseline
            baseline = float(self._explainer.expected_value)
            if isinstance(baseline, (list, np.ndarray)):
                baseline = float(baseline[0])

            # Generate summary
            summary = self._generate_summary(contributions[:3], prediction_value)

            # Calculate provenance
            input_hash = hashlib.sha256(
                feature_vector.tobytes()
            ).hexdigest()[:16]

            provenance_data = {
                "explanation_id": explanation_id,
                "prediction_id": prediction_id,
                "model_id": self.model_id,
                "input_hash": input_hash,
                "timestamp": timestamp.isoformat(),
            }
            provenance_hash = hashlib.sha256(
                json.dumps(provenance_data, sort_keys=True).encode()
            ).hexdigest()

            return ModelExplanation(
                explanation_id=explanation_id,
                prediction_id=prediction_id,
                model_id=self.model_id,
                model_version=self.model_version,
                timestamp=timestamp,
                prediction_value=prediction_value,
                prediction_label="anomaly" if prediction_value > 0.5 else "normal",
                confidence=abs(prediction_value - 0.5) * 2,
                method=ExplainabilityMethod.SHAP_TREE,
                baseline_value=baseline,
                feature_contributions=contributions,
                shap_values=shap_dict,
                summary_text=summary,
                detailed_explanation=self._generate_detailed(contributions),
                provenance_hash=provenance_hash,
                input_hash=input_hash,
            )

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._permutation_explain(
                feature_vector, prediction_value, prediction_id
            )

    def _permutation_explain(
        self,
        feature_vector: np.ndarray,
        prediction_value: float,
        prediction_id: str,
    ) -> ModelExplanation:
        """Fallback permutation-based explanation."""
        explanation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        # Get base prediction
        X = feature_vector.reshape(1, -1)

        if hasattr(self.model, 'decision_function'):
            base_score = self.model.decision_function(X)[0]
        elif hasattr(self.model, 'predict_proba'):
            base_score = self.model.predict_proba(X)[0, 1]
        else:
            base_score = prediction_value

        # Calculate permutation importance
        importances = {}
        for i, name in enumerate(self.feature_names):
            X_permuted = X.copy()
            X_permuted[0, i] = 0.0  # Zero out feature

            if hasattr(self.model, 'decision_function'):
                permuted_score = self.model.decision_function(X_permuted)[0]
            elif hasattr(self.model, 'predict_proba'):
                permuted_score = self.model.predict_proba(X_permuted)[0, 1]
            else:
                permuted_score = base_score

            importances[name] = float(base_score - permuted_score)

        contributions = self._build_contributions(importances, feature_vector)

        return ModelExplanation(
            explanation_id=explanation_id,
            prediction_id=prediction_id,
            model_id=self.model_id,
            model_version=self.model_version,
            timestamp=timestamp,
            prediction_value=prediction_value,
            prediction_label="anomaly" if prediction_value > 0.5 else "normal",
            confidence=abs(prediction_value - 0.5) * 2,
            method=ExplainabilityMethod.PERMUTATION,
            baseline_value=0.0,
            feature_contributions=contributions,
            shap_values=importances,
            summary_text=self._generate_summary(contributions[:3], prediction_value),
            detailed_explanation=self._generate_detailed(contributions),
            provenance_hash=hashlib.sha256(
                f"{explanation_id}|{prediction_id}".encode()
            ).hexdigest(),
            input_hash=hashlib.sha256(feature_vector.tobytes()).hexdigest()[:16],
        )

    def _build_contributions(
        self,
        values: Dict[str, float],
        feature_vector: np.ndarray,
    ) -> List[FeatureContribution]:
        """Build sorted feature contributions."""
        # Sort by absolute value
        sorted_items = sorted(
            values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        total = sum(abs(v) for v in values.values()) or 1.0

        contributions = []
        for rank, (name, value) in enumerate(sorted_items, 1):
            idx = self.feature_names.index(name) if name in self.feature_names else 0
            feature_val = float(feature_vector[idx]) if idx < len(feature_vector) else 0.0

            direction = (
                ContributionDirection.POSITIVE if value > 0
                else ContributionDirection.NEGATIVE if value < 0
                else ContributionDirection.NEUTRAL
            )

            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=feature_val,
                contribution_value=value,
                contribution_direction=direction,
                contribution_percent=(abs(value) / total) * 100,
                rank=rank,
                description=self._describe_feature(name, feature_val, value),
            ))

        return contributions

    def _describe_feature(
        self,
        name: str,
        value: float,
        contribution: float
    ) -> str:
        """Generate human-readable feature description."""
        direction = "increases" if contribution > 0 else "decreases"
        return f"{name}={value:.2f} {direction} anomaly score by {abs(contribution):.4f}"

    def _generate_summary(
        self,
        top_contributions: List[FeatureContribution],
        prediction: float
    ) -> str:
        """Generate summary explanation for human review."""
        if not top_contributions:
            return "No significant contributing factors identified."

        factors = ", ".join([
            f"{c.feature_name} ({c.contribution_direction.value})"
            for c in top_contributions
        ])

        if prediction > 0.7:
            return f"High anomaly score driven by: {factors}"
        elif prediction > 0.5:
            return f"Moderate anomaly score. Key factors: {factors}"
        else:
            return f"Low anomaly score. Minor factors: {factors}"

    def _generate_detailed(
        self,
        contributions: List[FeatureContribution]
    ) -> str:
        """Generate detailed explanation."""
        lines = ["Feature Contribution Analysis:"]
        for c in contributions[:10]:
            lines.append(
                f"  {c.rank}. {c.feature_name}: {c.contribution_value:+.4f} "
                f"({c.contribution_percent:.1f}%) [{c.contribution_direction.value}]"
            )
        return "\n".join(lines)


class LIMEExplainer:
    """
    LIME-based local interpretable explanations.

    Creates locally faithful linear approximations for any model,
    providing intuitive feature importance for individual predictions.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_id: str = "unknown",
        model_version: str = "1.0.0",
    ):
        self.model = model
        self.feature_names = feature_names
        self.model_id = model_id
        self.model_version = model_version
        self._explainer = None
        self._is_initialized = False

    def initialize(self, training_data: np.ndarray) -> bool:
        """Initialize LIME explainer."""
        try:
            import lime
            import lime.lime_tabular
        except ImportError:
            logger.warning("LIME not available")
            return False

        try:
            self._explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=self.feature_names,
                mode='regression',
                discretize_continuous=True,
            )
            self._is_initialized = True
            logger.info("Initialized LIME explainer")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LIME: {e}")
            return False

    def explain(
        self,
        feature_vector: np.ndarray,
        prediction_value: float,
        prediction_id: str,
        num_features: int = 10,
    ) -> ModelExplanation:
        """Generate LIME explanation."""
        explanation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        if not self._is_initialized or self._explainer is None:
            # Return empty explanation
            return self._empty_explanation(
                explanation_id, prediction_id, prediction_value, timestamp
            )

        try:
            # Get prediction function
            if hasattr(self.model, 'decision_function'):
                predict_fn = self.model.decision_function
            elif hasattr(self.model, 'predict_proba'):
                predict_fn = lambda x: self.model.predict_proba(x)[:, 1]
            else:
                predict_fn = self.model.predict

            # Generate LIME explanation
            exp = self._explainer.explain_instance(
                feature_vector,
                predict_fn,
                num_features=num_features,
            )

            # Extract weights
            lime_weights = dict(exp.as_list())

            # Map to feature names
            weight_dict = {}
            for name_val, weight in lime_weights.items():
                # LIME returns "feature <= value" format
                for fn in self.feature_names:
                    if fn in name_val:
                        weight_dict[fn] = weight
                        break

            contributions = self._build_contributions(
                weight_dict, feature_vector
            )

            return ModelExplanation(
                explanation_id=explanation_id,
                prediction_id=prediction_id,
                model_id=self.model_id,
                model_version=self.model_version,
                timestamp=timestamp,
                prediction_value=prediction_value,
                prediction_label="anomaly" if prediction_value > 0.5 else "normal",
                confidence=abs(prediction_value - 0.5) * 2,
                method=ExplainabilityMethod.LIME,
                baseline_value=exp.intercept[0] if hasattr(exp, 'intercept') else 0.0,
                feature_contributions=contributions,
                lime_weights=weight_dict,
                summary_text=self._generate_summary(contributions[:3], prediction_value),
                detailed_explanation=self._generate_detailed(contributions),
                provenance_hash=hashlib.sha256(
                    f"{explanation_id}|{prediction_id}".encode()
                ).hexdigest(),
                input_hash=hashlib.sha256(feature_vector.tobytes()).hexdigest()[:16],
            )

        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return self._empty_explanation(
                explanation_id, prediction_id, prediction_value, timestamp
            )

    def _build_contributions(
        self,
        weights: Dict[str, float],
        feature_vector: np.ndarray,
    ) -> List[FeatureContribution]:
        """Build feature contributions from LIME weights."""
        sorted_items = sorted(
            weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        total = sum(abs(v) for v in weights.values()) or 1.0

        contributions = []
        for rank, (name, weight) in enumerate(sorted_items, 1):
            idx = self.feature_names.index(name) if name in self.feature_names else 0
            feature_val = float(feature_vector[idx]) if idx < len(feature_vector) else 0.0

            direction = (
                ContributionDirection.POSITIVE if weight > 0
                else ContributionDirection.NEGATIVE if weight < 0
                else ContributionDirection.NEUTRAL
            )

            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=feature_val,
                contribution_value=weight,
                contribution_direction=direction,
                contribution_percent=(abs(weight) / total) * 100,
                rank=rank,
            ))

        return contributions

    def _generate_summary(
        self,
        top_contributions: List[FeatureContribution],
        prediction: float
    ) -> str:
        """Generate summary."""
        if not top_contributions:
            return "No significant factors identified."

        factors = [c.feature_name for c in top_contributions]
        return f"Key factors: {', '.join(factors)}"

    def _generate_detailed(
        self,
        contributions: List[FeatureContribution]
    ) -> str:
        """Generate detailed explanation."""
        lines = ["LIME Local Explanation:"]
        for c in contributions[:10]:
            lines.append(f"  {c.rank}. {c.feature_name}: {c.contribution_value:+.4f}")
        return "\n".join(lines)

    def _empty_explanation(
        self,
        explanation_id: str,
        prediction_id: str,
        prediction_value: float,
        timestamp: datetime,
    ) -> ModelExplanation:
        """Return empty explanation when LIME unavailable."""
        return ModelExplanation(
            explanation_id=explanation_id,
            prediction_id=prediction_id,
            model_id=self.model_id,
            model_version=self.model_version,
            timestamp=timestamp,
            prediction_value=prediction_value,
            prediction_label="anomaly" if prediction_value > 0.5 else "normal",
            confidence=0.5,
            method=ExplainabilityMethod.LIME,
            baseline_value=0.0,
            feature_contributions=[],
            summary_text="LIME explanation unavailable",
            detailed_explanation="",
            provenance_hash=hashlib.sha256(
                f"{explanation_id}".encode()
            ).hexdigest(),
            input_hash="",
        )


class FugitiveExplainer:
    """
    Unified explainer for fugitive emissions anomaly detection.

    Combines SHAP and LIME with automatic fallback and caching.
    Designed for regulatory compliance with full audit trails.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_id: str = "fugitive_detector",
        model_version: str = "1.0.0",
        use_shap: bool = True,
        use_lime: bool = True,
    ):
        self.model = model
        self.feature_names = feature_names
        self.model_id = model_id
        self.model_version = model_version

        self._shap_explainer: Optional[SHAPExplainer] = None
        self._lime_explainer: Optional[LIMEExplainer] = None
        self._is_initialized = False
        self._explanation_cache: Dict[str, ModelExplanation] = {}
        self._cache_max_size = 1000

        if use_shap:
            self._shap_explainer = SHAPExplainer(
                model, feature_names, model_id, model_version
            )

        if use_lime:
            self._lime_explainer = LIMEExplainer(
                model, feature_names, model_id, model_version
            )

        logger.info(
            f"FugitiveExplainer initialized: SHAP={use_shap}, LIME={use_lime}"
        )

    def initialize(self, training_data: np.ndarray) -> bool:
        """Initialize explainers with training data."""
        success = False

        if self._shap_explainer is not None:
            # Use subsample for background data
            n_samples = min(100, len(training_data))
            indices = np.random.choice(
                len(training_data), n_samples, replace=False
            )
            background = training_data[indices]

            if self._shap_explainer.initialize(background):
                success = True
                logger.info("SHAP explainer initialized")

        if self._lime_explainer is not None:
            if self._lime_explainer.initialize(training_data):
                success = True
                logger.info("LIME explainer initialized")

        self._is_initialized = success
        return success

    def explain(
        self,
        feature_vector: np.ndarray,
        prediction_value: float,
        prediction_id: str,
        method: ExplainabilityMethod = ExplainabilityMethod.SHAP_TREE,
    ) -> ModelExplanation:
        """
        Generate explanation for a prediction.

        Args:
            feature_vector: Input features
            prediction_value: Model prediction
            prediction_id: Unique identifier
            method: Preferred explanation method

        Returns:
            ModelExplanation with feature contributions
        """
        # Check cache
        cache_key = f"{prediction_id}_{method.value}"
        if cache_key in self._explanation_cache:
            logger.debug(f"Cache hit for {cache_key}")
            return self._explanation_cache[cache_key]

        # Generate explanation
        if method in [ExplainabilityMethod.SHAP_TREE, ExplainabilityMethod.SHAP_KERNEL]:
            if self._shap_explainer is not None:
                explanation = self._shap_explainer.explain(
                    feature_vector, prediction_value, prediction_id
                )
            elif self._lime_explainer is not None:
                # Fallback to LIME
                explanation = self._lime_explainer.explain(
                    feature_vector, prediction_value, prediction_id
                )
            else:
                explanation = self._statistical_explain(
                    feature_vector, prediction_value, prediction_id
                )
        elif method == ExplainabilityMethod.LIME:
            if self._lime_explainer is not None:
                explanation = self._lime_explainer.explain(
                    feature_vector, prediction_value, prediction_id
                )
            elif self._shap_explainer is not None:
                explanation = self._shap_explainer.explain(
                    feature_vector, prediction_value, prediction_id
                )
            else:
                explanation = self._statistical_explain(
                    feature_vector, prediction_value, prediction_id
                )
        else:
            explanation = self._statistical_explain(
                feature_vector, prediction_value, prediction_id
            )

        # Cache result
        if len(self._explanation_cache) >= self._cache_max_size:
            # Remove oldest entries
            keys = list(self._explanation_cache.keys())[:100]
            for k in keys:
                del self._explanation_cache[k]

        self._explanation_cache[cache_key] = explanation

        return explanation

    def explain_batch(
        self,
        feature_vectors: np.ndarray,
        predictions: np.ndarray,
        prediction_ids: List[str],
        method: ExplainabilityMethod = ExplainabilityMethod.SHAP_TREE,
    ) -> List[ModelExplanation]:
        """Explain a batch of predictions."""
        return [
            self.explain(fv, float(pred), pid, method)
            for fv, pred, pid in zip(feature_vectors, predictions, prediction_ids)
        ]

    def _statistical_explain(
        self,
        feature_vector: np.ndarray,
        prediction_value: float,
        prediction_id: str,
    ) -> ModelExplanation:
        """Fallback statistical explanation based on z-scores."""
        explanation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        # Calculate z-score based contributions
        mean = np.mean(feature_vector)
        std = np.std(feature_vector) or 1.0

        contributions = []
        total_zscore = 0.0
        zscores = {}

        for i, name in enumerate(self.feature_names):
            if i < len(feature_vector):
                zscore = (feature_vector[i] - mean) / std
                zscores[name] = zscore
                total_zscore += abs(zscore)

        for rank, (name, zscore) in enumerate(
            sorted(zscores.items(), key=lambda x: abs(x[1]), reverse=True), 1
        ):
            idx = self.feature_names.index(name)
            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=float(feature_vector[idx]) if idx < len(feature_vector) else 0.0,
                contribution_value=zscore,
                contribution_direction=(
                    ContributionDirection.POSITIVE if zscore > 0
                    else ContributionDirection.NEGATIVE if zscore < 0
                    else ContributionDirection.NEUTRAL
                ),
                contribution_percent=(abs(zscore) / total_zscore * 100) if total_zscore > 0 else 0,
                rank=rank,
            ))

        return ModelExplanation(
            explanation_id=explanation_id,
            prediction_id=prediction_id,
            model_id=self.model_id,
            model_version=self.model_version,
            timestamp=timestamp,
            prediction_value=prediction_value,
            prediction_label="anomaly" if prediction_value > 0.5 else "normal",
            confidence=abs(prediction_value - 0.5) * 2,
            method=ExplainabilityMethod.STATISTICAL,
            baseline_value=0.0,
            feature_contributions=contributions,
            summary_text=f"Statistical analysis: {len(contributions)} features ranked",
            detailed_explanation="Based on z-score analysis",
            provenance_hash=hashlib.sha256(
                f"{explanation_id}|{prediction_id}".encode()
            ).hexdigest(),
            input_hash=hashlib.sha256(feature_vector.tobytes()).hexdigest()[:16],
        )

    def clear_cache(self) -> int:
        """Clear explanation cache."""
        count = len(self._explanation_cache)
        self._explanation_cache.clear()
        return count


# Export all public classes
__all__ = [
    "ExplainabilityMethod",
    "ContributionDirection",
    "FeatureContribution",
    "ModelExplanation",
    "SHAPExplainer",
    "LIMEExplainer",
    "FugitiveExplainer",
]
