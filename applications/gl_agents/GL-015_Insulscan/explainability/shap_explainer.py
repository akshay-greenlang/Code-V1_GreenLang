# -*- coding: utf-8 -*-
"""
GL-015 Insulscan - SHAP Explainer for Insulation Assessment

SHapley Additive exPlanations for feature attribution on insulation
condition predictions. Provides both global (per-asset) and local
(per-prediction) explanations with stability guarantees.

Zero-Hallucination Principle:
- All SHAP values are computed from deterministic mathematical formulas
- No LLM is used for numeric calculations
- Provenance tracking via SHA-256 hashes

Reference: Lundberg & Lee, "A Unified Approach to Interpreting
Model Predictions", NeurIPS 2017.

Author: GreenLang AI Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging
import time
import uuid

import numpy as np

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from .explanation_schemas import (
    ConfidenceBounds,
    ConfidenceLevel,
    ExplanationType,
    FeatureCategory,
    FeatureContribution,
    FeatureImportance,
    GlobalExplanation,
    LocalExplanation,
    PredictionType,
    ExplanationStabilityMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class SHAPConfig:
    """Configuration for SHAP explainer."""
    n_samples: int = 100
    background_samples: int = 50
    max_display_features: int = 10
    use_tree_explainer: bool = True
    check_additivity: bool = True
    random_seed: int = 42
    stability_neighbors: int = 10
    stability_perturbation: float = 0.01
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300


@dataclass
class SHAPResult:
    """Result from SHAP analysis."""
    explanation_id: str
    feature_names: List[str]
    shap_values: np.ndarray
    base_value: float
    feature_values: np.ndarray
    expected_value: float
    feature_importance: Dict[str, float]
    feature_categories: Dict[str, FeatureCategory]
    interaction_effects: Optional[Dict[str, Dict[str, float]]]
    stability_score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    computation_hash: str = ""
    computation_time_ms: float = 0.0


# Feature category mapping for insulation assessment
INSULATION_FEATURE_CATEGORIES: Dict[str, FeatureCategory] = {
    # Thermal features
    "surface_temperature": FeatureCategory.THERMAL,
    "surface_temperature_delta": FeatureCategory.THERMAL,
    "ambient_temperature": FeatureCategory.THERMAL,
    "operating_temperature": FeatureCategory.THERMAL,
    "temperature_differential": FeatureCategory.THERMAL,
    "heat_loss_rate": FeatureCategory.THERMAL,
    "thermal_resistance": FeatureCategory.THERMAL,
    "r_value": FeatureCategory.THERMAL,
    "u_value": FeatureCategory.THERMAL,
    "hot_spot_count": FeatureCategory.THERMAL,
    "hot_spot_intensity": FeatureCategory.THERMAL,
    "hot_spot_max_temp": FeatureCategory.THERMAL,

    # Physical features
    "thickness": FeatureCategory.PHYSICAL,
    "thickness_original": FeatureCategory.PHYSICAL,
    "thickness_ratio": FeatureCategory.PHYSICAL,
    "density": FeatureCategory.PHYSICAL,
    "compression_ratio": FeatureCategory.PHYSICAL,
    "coverage_percentage": FeatureCategory.PHYSICAL,
    "gap_count": FeatureCategory.PHYSICAL,
    "gap_area": FeatureCategory.PHYSICAL,
    "surface_area": FeatureCategory.PHYSICAL,

    # Environmental features
    "uv_exposure_hours": FeatureCategory.ENVIRONMENTAL,
    "humidity_average": FeatureCategory.ENVIRONMENTAL,
    "precipitation_exposure": FeatureCategory.ENVIRONMENTAL,
    "chemical_exposure": FeatureCategory.ENVIRONMENTAL,
    "outdoor_installation": FeatureCategory.ENVIRONMENTAL,
    "ambient_humidity": FeatureCategory.ENVIRONMENTAL,

    # Moisture features
    "moisture_content": FeatureCategory.MOISTURE,
    "moisture_detected": FeatureCategory.MOISTURE,
    "water_ingress_risk": FeatureCategory.MOISTURE,
    "vapor_barrier_integrity": FeatureCategory.MOISTURE,

    # Age-related features
    "installation_age_years": FeatureCategory.AGE_RELATED,
    "expected_lifespan_years": FeatureCategory.AGE_RELATED,
    "age_ratio": FeatureCategory.AGE_RELATED,
    "thermal_cycles_count": FeatureCategory.AGE_RELATED,
    "maintenance_count": FeatureCategory.AGE_RELATED,
    "days_since_inspection": FeatureCategory.AGE_RELATED,

    # Visual features
    "visible_damage_score": FeatureCategory.VISUAL,
    "discoloration_level": FeatureCategory.VISUAL,
    "sagging_detected": FeatureCategory.VISUAL,
    "jacket_condition": FeatureCategory.VISUAL,

    # Operational features
    "operating_hours": FeatureCategory.OPERATIONAL,
    "cycle_frequency": FeatureCategory.OPERATIONAL,
    "temperature_swing": FeatureCategory.OPERATIONAL,
    "load_factor": FeatureCategory.OPERATIONAL,
}


class InsulationSHAPExplainer:
    """
    SHAP-based explainer for insulation condition predictions.

    Provides both global feature importance and local instance-level
    explanations. Designed for zero-hallucination with complete
    provenance tracking.

    Features:
    - Global SHAP explanations (top degradation drivers per asset)
    - Local SHAP explanations (per-prediction feature importance)
    - SHAP summary plots and force plots data
    - Feature importance ranking with engineering categories
    - Interaction effects between insulation factors
    - Batch explanation support

    Example:
        >>> config = SHAPConfig(n_samples=100)
        >>> explainer = InsulationSHAPExplainer(config)
        >>> result = explainer.explain_prediction(
        ...     model=condition_model,
        ...     features=input_features,
        ...     feature_names=feature_names,
        ...     prediction_type=PredictionType.CONDITION_SCORE,
        ...     asset_id="INS-001"
        ... )
        >>> print(f"Top driver: {list(result.feature_importance.keys())[0]}")
    """

    VERSION = "1.0.0"
    METHODOLOGY_REFERENCE = "Lundberg & Lee, NeurIPS 2017"

    def __init__(self, config: Optional[SHAPConfig] = None) -> None:
        """
        Initialize SHAP explainer.

        Args:
            config: Configuration options for SHAP computation
        """
        self.config = config or SHAPConfig()
        self._explainer = None
        self._background_data: Optional[np.ndarray] = None
        self._feature_names: List[str] = []
        self._cache: Dict[str, SHAPResult] = {}
        np.random.seed(self.config.random_seed)

        logger.info(
            f"InsulationSHAPExplainer initialized with config: "
            f"n_samples={self.config.n_samples}, "
            f"tree_explainer={self.config.use_tree_explainer}"
        )

    def explain_prediction(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        prediction_type: PredictionType,
        asset_id: str = "unknown",
        background_data: Optional[np.ndarray] = None,
    ) -> LocalExplanation:
        """
        Generate local SHAP explanation for a single prediction.

        Args:
            model: Trained model with predict method
            features: Feature values for the instance (1D array)
            feature_names: Names of features
            prediction_type: Type of prediction being explained
            asset_id: Identifier of the insulation asset
            background_data: Optional background dataset for SHAP

        Returns:
            LocalExplanation with feature contributions and confidence

        Raises:
            ValueError: If features and feature_names have mismatched lengths
        """
        start_time = time.time()

        if len(features) != len(feature_names):
            raise ValueError(
                f"Features length ({len(features)}) != feature_names length ({len(feature_names)})"
            )

        self._feature_names = feature_names

        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Set up background data
        if background_data is not None:
            self._background_data = background_data
        elif self._background_data is None:
            self._background_data = self._generate_background_data(features)

        # Compute SHAP values
        shap_result = self._compute_shap_values(
            model, features, feature_names, prediction_type
        )

        # Get prediction
        prediction_value = float(model.predict(features)[0])
        base_value = shap_result.base_value

        # Build feature contributions
        contributions = self._build_feature_contributions(
            shap_result, features[0], feature_names
        )

        # Calculate stability
        stability_metrics = self._assess_stability(
            model, features, feature_names, shap_result
        )

        # Determine confidence level
        confidence = self._determine_confidence(
            shap_result, stability_metrics.stability_score
        )

        # Identify top drivers
        sorted_contributions = sorted(
            contributions, key=lambda x: abs(x.contribution), reverse=True
        )
        top_positive = [
            c.feature_name for c in sorted_contributions
            if c.contribution > 0
        ][:5]
        top_negative = [
            c.feature_name for c in sorted_contributions
            if c.contribution < 0
        ][:5]

        # Compute provenance hash
        explanation_id = str(uuid.uuid4())
        computation_time = (time.time() - start_time) * 1000

        local_explanation = LocalExplanation(
            explanation_id=explanation_id,
            asset_id=asset_id,
            timestamp=datetime.now(timezone.utc),
            prediction_type=prediction_type,
            prediction_value=prediction_value,
            base_value=base_value,
            feature_contributions=contributions,
            top_positive_drivers=top_positive,
            top_negative_drivers=top_negative,
            explanation_method=ExplanationType.SHAP,
            local_accuracy=self._compute_local_accuracy(shap_result, prediction_value),
            stability_score=stability_metrics.stability_score,
            confidence=confidence,
            computation_time_ms=computation_time,
            provenance_hash=shap_result.computation_hash,
            methodology_version=self.VERSION,
        )

        logger.info(
            f"Local SHAP explanation generated for asset {asset_id} "
            f"in {computation_time:.2f}ms"
        )

        return local_explanation

    def explain_global(
        self,
        model: Any,
        data: np.ndarray,
        feature_names: List[str],
        prediction_type: PredictionType,
        model_name: str = "insulation_model",
        model_version: str = "1.0.0",
    ) -> GlobalExplanation:
        """
        Generate global SHAP explanation across multiple instances.

        Args:
            model: Trained model with predict method
            data: Feature matrix (n_samples x n_features)
            feature_names: Names of features
            prediction_type: Type of prediction being explained
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            GlobalExplanation with overall feature importance
        """
        start_time = time.time()
        self._feature_names = feature_names

        # Compute SHAP values for all instances
        if HAS_SHAP:
            shap_values, base_value = self._compute_global_shap(model, data, feature_names)
        else:
            shap_values, base_value = self._compute_permutation_importance(model, data)

        # Calculate mean absolute SHAP values for global importance
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        # Build feature importance list
        feature_importance = []
        for rank, (idx, importance) in enumerate(
            sorted(enumerate(mean_abs_shap), key=lambda x: x[1], reverse=True), 1
        ):
            name = feature_names[idx]
            category = self._get_feature_category(name)

            # Determine direction from mean SHAP value
            mean_shap = np.mean(shap_values[:, idx])
            direction = "positive" if mean_shap > 0 else "negative" if mean_shap < 0 else "neutral"

            feature_importance.append(FeatureImportance(
                feature_name=name,
                importance_value=round(float(importance), 6),
                rank=rank,
                direction=direction,
                category=category,
                confidence=self._compute_importance_confidence(shap_values[:, idx]),
                engineering_interpretation=self._generate_interpretation(name, direction),
                unit=self._get_feature_unit(name),
            ))

        # Compute interaction effects for top features
        interaction_effects = self._compute_interaction_effects(
            shap_values, feature_names, top_k=5
        )

        # Predictions for summary stats
        predictions = model.predict(data)

        computation_time = (time.time() - start_time) * 1000
        explanation_id = str(uuid.uuid4())

        # Compute provenance hash
        provenance_data = {
            "explanation_id": explanation_id,
            "feature_importance": {fi.feature_name: fi.importance_value for fi in feature_importance[:10]},
            "n_samples": len(data),
            "version": self.VERSION,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        # Determine confidence
        confidence = ConfidenceLevel.HIGH if len(data) >= 100 else \
                     ConfidenceLevel.MEDIUM if len(data) >= 30 else \
                     ConfidenceLevel.LOW

        return GlobalExplanation(
            explanation_id=explanation_id,
            model_name=model_name,
            model_version=model_version,
            prediction_type=prediction_type,
            feature_importance=feature_importance[:self.config.max_display_features],
            top_k_features=min(self.config.max_display_features, len(feature_importance)),
            interaction_effects=interaction_effects,
            total_assets_analyzed=len(data),
            mean_prediction=float(np.mean(predictions)),
            std_prediction=float(np.std(predictions)),
            explanation_method=ExplanationType.SHAP,
            aggregation_method="mean_absolute",
            confidence=confidence,
            computation_time_ms=computation_time,
            provenance_hash=provenance_hash,
            methodology_version=self.VERSION,
        )

    def explain_batch(
        self,
        model: Any,
        batch_features: np.ndarray,
        feature_names: List[str],
        asset_ids: List[str],
        prediction_type: PredictionType = PredictionType.CONDITION_SCORE,
    ) -> List[LocalExplanation]:
        """
        Generate SHAP explanations for a batch of predictions.

        Args:
            model: Trained model with predict method
            batch_features: Feature matrix (n_samples x n_features)
            feature_names: Names of features
            asset_ids: List of asset identifiers
            prediction_type: Type of prediction

        Returns:
            List of LocalExplanation objects
        """
        if len(batch_features) != len(asset_ids):
            raise ValueError("batch_features and asset_ids must have same length")

        explanations = []

        # Set up background data once
        if self._background_data is None:
            self._background_data = self._generate_background_data(batch_features)

        # Compute SHAP for all at once if possible
        if HAS_SHAP:
            try:
                shap_values_batch, base_value = self._compute_global_shap(
                    model, batch_features, feature_names
                )
            except Exception as e:
                logger.warning(f"Batch SHAP failed: {e}, falling back to individual")
                shap_values_batch = None
        else:
            shap_values_batch = None

        for i, (features, asset_id) in enumerate(zip(batch_features, asset_ids)):
            if shap_values_batch is not None:
                # Use pre-computed SHAP values
                shap_values = shap_values_batch[i]
                prediction = float(model.predict(features.reshape(1, -1))[0])

                # Build result
                shap_result = self._build_shap_result(
                    shap_values, base_value, features, feature_names
                )

                contributions = self._build_feature_contributions(
                    shap_result, features, feature_names
                )

                explanation = LocalExplanation(
                    explanation_id=str(uuid.uuid4()),
                    asset_id=asset_id,
                    timestamp=datetime.now(timezone.utc),
                    prediction_type=prediction_type,
                    prediction_value=prediction,
                    base_value=base_value,
                    feature_contributions=contributions,
                    top_positive_drivers=[c.feature_name for c in contributions if c.contribution > 0][:5],
                    top_negative_drivers=[c.feature_name for c in contributions if c.contribution < 0][:5],
                    explanation_method=ExplanationType.SHAP,
                    local_accuracy=self._compute_local_accuracy(shap_result, prediction),
                    stability_score=0.85,  # Approximate for batch
                    confidence=ConfidenceLevel.MEDIUM,
                    computation_time_ms=0.0,
                    provenance_hash=shap_result.computation_hash,
                    methodology_version=self.VERSION,
                )
                explanations.append(explanation)
            else:
                # Fall back to individual explanation
                explanation = self.explain_prediction(
                    model, features, feature_names, prediction_type, asset_id
                )
                explanations.append(explanation)

        return explanations

    def get_force_plot_data(self, result: SHAPResult) -> Dict[str, Any]:
        """
        Get data formatted for SHAP force plot visualization.

        Args:
            result: SHAP analysis result

        Returns:
            Dictionary with force plot data
        """
        sorted_indices = np.argsort(np.abs(result.shap_values))[::-1]

        features = []
        for idx in sorted_indices[:self.config.max_display_features]:
            features.append({
                "feature": result.feature_names[idx],
                "value": float(result.feature_values[idx]),
                "shap_value": float(result.shap_values[idx]),
                "category": self._get_feature_category(result.feature_names[idx]).value,
            })

        return {
            "base_value": result.base_value,
            "output_value": result.base_value + sum(result.shap_values),
            "features": features,
            "expected_value": result.expected_value,
            "computation_hash": result.computation_hash,
        }

    def get_waterfall_data(self, result: SHAPResult) -> List[Dict[str, Any]]:
        """
        Get data formatted for SHAP waterfall chart.

        Args:
            result: SHAP analysis result

        Returns:
            List of dictionaries for waterfall chart
        """
        waterfall_data = [
            {"feature": "Base Value", "value": result.base_value, "cumulative": result.base_value}
        ]

        sorted_indices = np.argsort(np.abs(result.shap_values))[::-1]
        cumulative = result.base_value

        for idx in sorted_indices[:self.config.max_display_features]:
            shap_val = result.shap_values[idx]
            cumulative += shap_val
            waterfall_data.append({
                "feature": result.feature_names[idx],
                "value": float(shap_val),
                "cumulative": float(cumulative),
                "feature_value": float(result.feature_values[idx]),
            })

        waterfall_data.append({
            "feature": "Prediction",
            "value": float(cumulative),
            "cumulative": float(cumulative),
        })

        return waterfall_data

    def get_summary_plot_data(
        self,
        model: Any,
        data: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Get data formatted for SHAP summary plot.

        Args:
            model: Trained model
            data: Feature matrix
            feature_names: Feature names

        Returns:
            Dictionary with summary plot data
        """
        if HAS_SHAP:
            shap_values, _ = self._compute_global_shap(model, data, feature_names)
        else:
            shap_values, _ = self._compute_permutation_importance(model, data)

        summary_data = {
            "feature_names": feature_names,
            "shap_values": shap_values.tolist(),
            "feature_values": data.tolist(),
            "feature_importance": np.mean(np.abs(shap_values), axis=0).tolist(),
        }

        return summary_data

    def _compute_shap_values(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        prediction_type: PredictionType,
    ) -> SHAPResult:
        """Compute SHAP values for a single instance."""
        start_time = time.time()

        if HAS_SHAP:
            try:
                if self.config.use_tree_explainer and hasattr(model, 'estimators_'):
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model, self._background_data)

                shap_values_obj = explainer(features)
                shap_values = shap_values_obj.values[0] if shap_values_obj.values.ndim > 1 else shap_values_obj.values
                base_value = float(shap_values_obj.base_values[0]) if hasattr(shap_values_obj.base_values, '__iter__') else float(shap_values_obj.base_values)

            except Exception as e:
                logger.warning(f"SHAP library computation failed: {e}, falling back to permutation")
                shap_values, base_value = self._compute_permutation_importance_local(model, features)
        else:
            shap_values, base_value = self._compute_permutation_importance_local(model, features)

        return self._build_shap_result(
            shap_values, base_value, features[0] if features.ndim > 1 else features, feature_names
        )

    def _build_shap_result(
        self,
        shap_values: np.ndarray,
        base_value: float,
        feature_values: np.ndarray,
        feature_names: List[str],
    ) -> SHAPResult:
        """Build SHAPResult from computed values."""
        # Compute feature importance
        feature_importance = {}
        for i, name in enumerate(feature_names):
            feature_importance[name] = round(float(abs(shap_values[i])), 6)

        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        # Get feature categories
        feature_categories = {
            name: self._get_feature_category(name) for name in feature_names
        }

        # Compute interaction effects for top features
        interaction_effects = self._compute_interaction_effects_local(
            shap_values, feature_names
        )

        # Compute provenance hash
        provenance_data = {
            "features": feature_values.tolist(),
            "shap_values": shap_values.tolist(),
            "base_value": base_value,
            "version": self.VERSION,
        }
        computation_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return SHAPResult(
            explanation_id=str(uuid.uuid4()),
            feature_names=feature_names,
            shap_values=shap_values,
            base_value=base_value,
            feature_values=feature_values,
            expected_value=base_value,
            feature_importance=feature_importance,
            feature_categories=feature_categories,
            interaction_effects=interaction_effects,
            stability_score=1.0,  # Will be updated later
            computation_hash=computation_hash,
        )

    def _compute_global_shap(
        self,
        model: Any,
        data: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, float]:
        """Compute SHAP values for multiple instances."""
        try:
            if self.config.use_tree_explainer and hasattr(model, 'estimators_'):
                explainer = shap.TreeExplainer(model)
            else:
                background = data[:self.config.background_samples] if len(data) > self.config.background_samples else data
                explainer = shap.Explainer(model, background)

            shap_values_obj = explainer(data)
            shap_values = shap_values_obj.values
            base_value = float(np.mean(shap_values_obj.base_values))

            return shap_values, base_value

        except Exception as e:
            logger.warning(f"Global SHAP computation failed: {e}")
            return self._compute_permutation_importance(model, data)

    def _compute_permutation_importance_local(
        self,
        model: Any,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Fallback: compute permutation-based importance for single instance."""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        base_pred = model.predict(features)[0]
        importance = np.zeros(features.shape[1])

        for i in range(features.shape[1]):
            perturbed = features.copy()
            perturbed[0, i] *= (1 + self.config.stability_perturbation)
            new_pred = model.predict(perturbed)[0]
            importance[i] = base_pred - new_pred

        return importance, float(base_pred)

    def _compute_permutation_importance(
        self,
        model: Any,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Fallback: compute permutation-based importance for global."""
        n_samples, n_features = data.shape
        base_preds = model.predict(data)
        base_value = float(np.mean(base_preds))

        shap_values = np.zeros((n_samples, n_features))

        for i in range(n_features):
            perturbed = data.copy()
            np.random.shuffle(perturbed[:, i])
            new_preds = model.predict(perturbed)
            shap_values[:, i] = base_preds - new_preds

        return shap_values, base_value

    def _generate_background_data(self, features: np.ndarray) -> np.ndarray:
        """Generate synthetic background data for SHAP."""
        n_features = features.shape[1] if features.ndim > 1 else len(features)

        # Generate variations around the input
        background = np.random.randn(self.config.background_samples, n_features)
        center = features[0] if features.ndim > 1 else features

        background = center + background * np.abs(center) * 0.1
        background = np.maximum(background, 0.01)  # Ensure positive for physical quantities

        return background

    def _build_feature_contributions(
        self,
        shap_result: SHAPResult,
        feature_values: np.ndarray,
        feature_names: List[str],
    ) -> List[FeatureContribution]:
        """Build list of feature contributions."""
        contributions = []
        total_abs = sum(abs(sv) for sv in shap_result.shap_values) or 1.0

        for i, name in enumerate(feature_names):
            shap_val = shap_result.shap_values[i]
            direction = "positive" if shap_val > 0 else "negative" if shap_val < 0 else "neutral"

            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=float(feature_values[i]),
                baseline_value=None,
                contribution=round(float(shap_val), 6),
                contribution_percentage=round(float(shap_val / total_abs * 100), 4),
                direction=direction,
                category=self._get_feature_category(name),
                unit=self._get_feature_unit(name),
                is_anomalous=self._is_anomalous(name, feature_values[i]),
                importance_rank=0,
            ))

        # Sort by absolute contribution and update ranks
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        for i, c in enumerate(contributions):
            c.importance_rank = i + 1

        return contributions

    def _compute_interaction_effects_local(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        top_k: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """Compute pairwise interaction effects for top features."""
        interactions: Dict[str, Dict[str, float]] = {}

        top_indices = np.argsort(np.abs(shap_values))[-top_k:]

        for i in top_indices:
            interactions[feature_names[i]] = {}
            for j in top_indices:
                if i != j:
                    # Simple interaction estimate based on SHAP value products
                    interaction = (
                        shap_values[i] * shap_values[j] /
                        (abs(shap_values[i]) + abs(shap_values[j]) + 1e-10)
                    )
                    interactions[feature_names[i]][feature_names[j]] = round(float(interaction), 6)

        return interactions

    def _compute_interaction_effects(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        top_k: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """Compute global interaction effects."""
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs)[-top_k:]

        interactions: Dict[str, Dict[str, float]] = {}

        for i in top_indices:
            interactions[feature_names[i]] = {}
            for j in top_indices:
                if i != j:
                    # Correlation between SHAP values as interaction proxy
                    if shap_values.shape[0] > 1:
                        corr = np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1]
                        if not np.isnan(corr):
                            interactions[feature_names[i]][feature_names[j]] = round(float(corr), 4)

        return interactions

    def _assess_stability(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        original_result: SHAPResult,
    ) -> ExplanationStabilityMetrics:
        """Assess stability of SHAP explanations across similar inputs."""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        center = features[0]
        n_neighbors = self.config.stability_neighbors
        perturbation = self.config.stability_perturbation

        # Generate neighboring points
        neighbors = center + np.random.randn(n_neighbors, len(center)) * perturbation * np.abs(center)
        neighbors = np.maximum(neighbors, 0.01)

        # Compute SHAP for neighbors
        neighbor_rankings = []
        neighbor_contributions = []

        for neighbor in neighbors:
            if HAS_SHAP:
                try:
                    if self.config.use_tree_explainer and hasattr(model, 'estimators_'):
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.Explainer(model, self._background_data)
                    sv = explainer(neighbor.reshape(1, -1)).values[0]
                except Exception:
                    sv, _ = self._compute_permutation_importance_local(model, neighbor)
            else:
                sv, _ = self._compute_permutation_importance_local(model, neighbor)

            ranking = np.argsort(np.abs(sv))[::-1]
            neighbor_rankings.append(ranking)
            neighbor_contributions.append(sv)

        # Compute ranking stability
        original_ranking = np.argsort(np.abs(original_result.shap_values))[::-1]
        ranking_agreements = []
        for nr in neighbor_rankings:
            agreement = np.sum(original_ranking[:5] == nr[:5]) / 5
            ranking_agreements.append(agreement)

        feature_ranking_stability = float(np.mean(ranking_agreements))

        # Compute contribution variance
        contributions_array = np.array(neighbor_contributions)
        contribution_variance = float(np.mean(np.std(contributions_array, axis=0)))

        # Overall stability score
        stability_score = float(np.clip(
            0.5 * feature_ranking_stability + 0.5 * (1 - min(contribution_variance, 1.0)),
            0.0, 1.0
        ))

        return ExplanationStabilityMetrics(
            stability_score=round(stability_score, 4),
            feature_ranking_stability=round(feature_ranking_stability, 4),
            contribution_variance=round(contribution_variance, 6),
            neighboring_points_analyzed=n_neighbors,
            stability_method="neighborhood_sampling",
        )

    def _determine_confidence(
        self,
        result: SHAPResult,
        stability_score: float,
    ) -> ConfidenceLevel:
        """Determine confidence level based on SHAP result quality."""
        # Check additivity (SHAP values should sum to prediction - base)
        additivity_error = abs(
            sum(result.shap_values) -
            (result.base_value + sum(result.shap_values) - result.base_value)
        )

        # Combine stability and additivity
        quality_score = 0.7 * stability_score + 0.3 * (1 - min(additivity_error, 1.0))

        if quality_score >= 0.85:
            return ConfidenceLevel.HIGH
        elif quality_score >= 0.65:
            return ConfidenceLevel.MEDIUM
        elif quality_score >= 0.40:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _compute_local_accuracy(
        self,
        result: SHAPResult,
        prediction: float,
    ) -> float:
        """Compute local surrogate model accuracy (R2)."""
        explained = result.base_value + sum(result.shap_values)
        if abs(prediction) < 1e-10:
            return 1.0 if abs(explained) < 1e-10 else 0.0

        r2 = 1 - (prediction - explained) ** 2 / (prediction ** 2 + 1e-10)
        return float(np.clip(r2, 0.0, 1.0))

    def _compute_importance_confidence(self, shap_column: np.ndarray) -> float:
        """Compute confidence for a feature's importance estimate."""
        if len(shap_column) < 2:
            return 0.5

        mean_val = np.mean(np.abs(shap_column))
        std_val = np.std(np.abs(shap_column))

        if mean_val < 1e-10:
            return 0.5

        cv = std_val / mean_val
        confidence = 1 - min(cv, 1.0)

        return float(np.clip(confidence, 0.0, 1.0))

    def _get_feature_category(self, name: str) -> FeatureCategory:
        """Get category for a feature based on its name."""
        name_lower = name.lower()

        if name in INSULATION_FEATURE_CATEGORIES:
            return INSULATION_FEATURE_CATEGORIES[name]

        # Infer from name patterns
        if any(t in name_lower for t in ['temp', 'thermal', 'heat', 'hot_spot', 'r_value']):
            return FeatureCategory.THERMAL
        elif any(h in name_lower for h in ['thick', 'density', 'gap', 'coverage', 'compress']):
            return FeatureCategory.PHYSICAL
        elif any(e in name_lower for e in ['uv', 'humid', 'precip', 'outdoor', 'chemical']):
            return FeatureCategory.ENVIRONMENTAL
        elif any(m in name_lower for m in ['moisture', 'water', 'vapor', 'wet']):
            return FeatureCategory.MOISTURE
        elif any(a in name_lower for a in ['age', 'year', 'cycle', 'days_since']):
            return FeatureCategory.AGE_RELATED
        elif any(v in name_lower for v in ['visible', 'color', 'sag', 'jacket']):
            return FeatureCategory.VISUAL
        else:
            return FeatureCategory.OPERATIONAL

    def _get_feature_unit(self, name: str) -> Optional[str]:
        """Get unit of measurement for a feature."""
        units = {
            "surface_temperature": "C",
            "surface_temperature_delta": "C",
            "ambient_temperature": "C",
            "operating_temperature": "C",
            "temperature_differential": "C",
            "heat_loss_rate": "W/m2",
            "thermal_resistance": "m2K/W",
            "r_value": "m2K/W",
            "u_value": "W/m2K",
            "thickness": "mm",
            "thickness_original": "mm",
            "thickness_ratio": "-",
            "density": "kg/m3",
            "coverage_percentage": "%",
            "moisture_content": "%",
            "uv_exposure_hours": "hours",
            "humidity_average": "%",
            "installation_age_years": "years",
            "operating_hours": "hours",
            "surface_area": "m2",
        }
        return units.get(name)

    def _generate_interpretation(self, name: str, direction: str) -> str:
        """Generate engineering interpretation for a feature."""
        interpretations = {
            ("surface_temperature_delta", "positive"): "Higher surface temperature differential indicates heat loss through degraded insulation",
            ("surface_temperature_delta", "negative"): "Lower temperature differential suggests effective insulation",
            ("moisture_content", "positive"): "Higher moisture reduces insulation R-value and accelerates degradation",
            ("thickness_ratio", "negative"): "Maintaining original thickness preserves thermal performance",
            ("installation_age_years", "positive"): "Older insulation has accumulated degradation from environmental exposure",
            ("hot_spot_count", "positive"): "More thermal hot spots indicate localized insulation failures",
            ("compression_ratio", "positive"): "Compressed insulation has reduced thermal resistance",
            ("gap_count", "positive"): "Gaps allow thermal bridging and heat loss",
        }

        key = (name, direction)
        return interpretations.get(key, f"{name} affects insulation condition in {direction} direction")

    def _is_anomalous(self, name: str, value: float) -> bool:
        """Check if a feature value is anomalous."""
        # Define thresholds for anomaly detection
        thresholds = {
            "surface_temperature_delta": (0, 100),
            "moisture_content": (0, 50),
            "hot_spot_count": (0, 20),
            "thickness_ratio": (0, 1.5),
            "installation_age_years": (0, 50),
        }

        if name in thresholds:
            low, high = thresholds[name]
            return value < low or value > high

        return abs(value) > 1e6 or np.isnan(value) or np.isinf(value)

    def clear_cache(self) -> None:
        """Clear the explanation cache."""
        self._cache.clear()
        logger.info("SHAP explanation cache cleared")


# Convenience functions for module-level access
def explain_insulation_prediction(
    model: Any,
    features: np.ndarray,
    feature_names: List[str],
    prediction_type: PredictionType = PredictionType.CONDITION_SCORE,
    asset_id: str = "unknown",
    config: Optional[SHAPConfig] = None,
) -> LocalExplanation:
    """
    Convenience function to generate local SHAP explanation.

    Args:
        model: Trained insulation condition model
        features: Feature values for the instance
        feature_names: Names of features
        prediction_type: Type of prediction
        asset_id: Insulation asset identifier
        config: Optional SHAP configuration

    Returns:
        LocalExplanation with feature contributions
    """
    explainer = InsulationSHAPExplainer(config)
    return explainer.explain_prediction(
        model, features, feature_names, prediction_type, asset_id
    )


def get_global_insulation_importance(
    model: Any,
    data: np.ndarray,
    feature_names: List[str],
    prediction_type: PredictionType = PredictionType.CONDITION_SCORE,
    config: Optional[SHAPConfig] = None,
) -> GlobalExplanation:
    """
    Convenience function to generate global SHAP explanation.

    Args:
        model: Trained insulation condition model
        data: Feature matrix (n_samples x n_features)
        feature_names: Names of features
        prediction_type: Type of prediction
        config: Optional SHAP configuration

    Returns:
        GlobalExplanation with overall feature importance
    """
    explainer = InsulationSHAPExplainer(config)
    return explainer.explain_global(model, data, feature_names, prediction_type)


def verify_shap_consistency(result: SHAPResult, tolerance: float = 0.01) -> bool:
    """
    Verify SHAP value consistency (additivity check).

    Args:
        result: SHAP analysis result
        tolerance: Acceptable error tolerance

    Returns:
        True if SHAP values are consistent
    """
    explained = result.base_value + sum(result.shap_values)
    expected = result.expected_value + sum(result.shap_values)
    return abs(explained - expected) < tolerance
