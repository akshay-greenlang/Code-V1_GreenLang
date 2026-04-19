"""
SHAPExplainer - SHAP-based model explanations for combustion optimization.

This module provides SHAP (SHapley Additive exPlanations) based explanations
for machine learning model predictions. SHAP values provide theoretically
grounded feature attributions based on cooperative game theory.

Example:
    >>> explainer = SHAPExplainer(config)
    >>> shap_values = explainer.compute_shap_values(model, features_df)
    >>> importance = explainer.get_feature_importance(shap_values)
"""

import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd

from .explainability_payload import (
    SHAPExplanation,
    SHAPValues,
    FeatureContribution,
    ImpactDirection,
    ExplanationType,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


class BaseModel(Protocol):
    """Protocol for models that can be explained with SHAP."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input features."""
        ...


class Figure(Protocol):
    """Protocol for matplotlib-like figure objects."""

    def savefig(self, fname: str, **kwargs) -> None:
        """Save figure to file."""
        ...


class SHAPExplainerConfig:
    """Configuration for SHAPExplainer."""

    def __init__(
        self,
        n_samples: int = 100,
        feature_perturbation: str = "interventional",
        check_additivity: bool = True,
        compute_interactions: bool = False,
        top_k_features: int = 10,
        confidence_threshold: float = 0.85,
        random_seed: int = 42,
    ):
        """
        Initialize SHAPExplainer configuration.

        Args:
            n_samples: Number of background samples for SHAP
            feature_perturbation: Perturbation method ("interventional" or "tree_path_dependent")
            check_additivity: Verify SHAP values sum to prediction
            compute_interactions: Whether to compute interaction values
            top_k_features: Number of top features to include in explanations
            confidence_threshold: Threshold for high confidence explanations
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.feature_perturbation = feature_perturbation
        self.check_additivity = check_additivity
        self.compute_interactions = compute_interactions
        self.top_k_features = top_k_features
        self.confidence_threshold = confidence_threshold
        self.random_seed = random_seed


class SHAPExplainer:
    """
    SHAP-based explainer for combustion optimization models.

    This class computes SHAP values for ML model predictions and generates
    human-readable explanations. SHAP values satisfy important properties:
    - Local accuracy: SHAP values sum to the difference from expected value
    - Missingness: Features with no impact have zero SHAP value
    - Consistency: Monotonic feature importance across models

    Attributes:
        config: Configuration parameters for SHAP computation
        background_data: Background dataset for SHAP calculations

    Example:
        >>> config = SHAPExplainerConfig(n_samples=100)
        >>> explainer = SHAPExplainer(config)
        >>> shap_values = explainer.compute_shap_values(model, features_df)
    """

    def __init__(
        self,
        config: Optional[SHAPExplainerConfig] = None,
        background_data: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize SHAPExplainer.

        Args:
            config: Configuration parameters. Uses defaults if not provided.
            background_data: Background dataset for SHAP calculations.
        """
        self.config = config or SHAPExplainerConfig()
        self.background_data = background_data
        self._shap_explainer = None
        np.random.seed(self.config.random_seed)
        logger.info("SHAPExplainer initialized")

    def set_background_data(self, data: pd.DataFrame) -> None:
        """
        Set background data for SHAP calculations.

        Args:
            data: DataFrame with background samples
        """
        self.background_data = data
        self._shap_explainer = None  # Reset explainer
        logger.info(f"Background data set with {len(data)} samples")

    def compute_shap_values(
        self,
        model: BaseModel,
        features: pd.DataFrame,
    ) -> SHAPValues:
        """
        Compute SHAP values for model predictions.

        Uses KernelSHAP algorithm to compute model-agnostic SHAP values.
        For tree-based models, considers using TreeSHAP for efficiency.

        Args:
            model: Trained model with predict method
            features: DataFrame with features to explain

        Returns:
            SHAPValues containing base value, SHAP values, and prediction

        Example:
            >>> shap_values = explainer.compute_shap_values(model, features_df)
            >>> print(shap_values.base_value)
            0.85
        """
        start_time = datetime.now()
        logger.info(f"Computing SHAP values for {len(features)} instances")

        # Ensure we have background data
        if self.background_data is None:
            logger.warning("No background data set, using input features as background")
            self.background_data = features

        # Get feature names
        feature_names = list(features.columns)

        # Sample background data if needed
        n_background = min(self.config.n_samples, len(self.background_data))
        background_sample = self.background_data.sample(
            n=n_background, random_state=self.config.random_seed
        )

        # Compute base value (expected prediction on background)
        base_predictions = model.predict(background_sample.values)
        base_value = float(np.mean(base_predictions))

        # Compute SHAP values using KernelSHAP approximation
        # This is a simplified implementation - production would use shap library
        shap_values_array = self._compute_kernel_shap(
            model, features.values, background_sample.values, base_value
        )

        # Get predictions for instances
        predictions = model.predict(features.values)

        # Average SHAP values if multiple instances
        if len(features) > 1:
            shap_values_list = list(np.mean(shap_values_array, axis=0))
            prediction = float(np.mean(predictions))
            feature_values = list(features.mean().values)
        else:
            shap_values_list = list(shap_values_array[0])
            prediction = float(predictions[0])
            feature_values = list(features.iloc[0].values)

        # Compute interaction values if requested
        interaction_values = None
        if self.config.compute_interactions:
            interaction_values = self._compute_interaction_values(
                shap_values_array, feature_names
            )

        # Verify additivity
        if self.config.check_additivity:
            expected_prediction = base_value + sum(shap_values_list)
            if abs(expected_prediction - prediction) > 0.01:
                logger.warning(
                    f"SHAP additivity check: expected {expected_prediction:.4f}, "
                    f"got {prediction:.4f}"
                )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"SHAP values computed in {processing_time:.1f}ms")

        return SHAPValues(
            base_value=round(base_value, 6),
            feature_names=feature_names,
            shap_values=[round(v, 6) for v in shap_values_list],
            feature_values=[round(v, 6) for v in feature_values],
            prediction=round(prediction, 6),
            interaction_values=interaction_values,
        )

    def get_feature_importance(
        self,
        shap_values: SHAPValues,
    ) -> Dict[str, float]:
        """
        Calculate global feature importance from SHAP values.

        Feature importance is computed as mean absolute SHAP value,
        providing a measure of average impact on predictions.

        Args:
            shap_values: Computed SHAP values

        Returns:
            Dictionary mapping feature names to importance scores

        Example:
            >>> importance = explainer.get_feature_importance(shap_values)
            >>> print(importance)
            {'o2_percent': 0.15, 'load_percent': 0.12, ...}
        """
        importance = {}
        for name, value in zip(shap_values.feature_names, shap_values.shap_values):
            importance[name] = round(abs(value), 6)

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        logger.debug(f"Feature importance computed for {len(importance)} features")
        return importance

    def explain_prediction(
        self,
        model: BaseModel,
        instance: Dict[str, float],
    ) -> SHAPExplanation:
        """
        Generate SHAP explanation for a single prediction.

        Computes SHAP values and creates a comprehensive explanation
        including feature contributions and interpretation.

        Args:
            model: Trained model to explain
            instance: Dictionary of feature values for single instance

        Returns:
            SHAPExplanation with complete analysis

        Example:
            >>> instance = {"o2_percent": 3.5, "load_percent": 75.0}
            >>> explanation = explainer.explain_prediction(model, instance)
        """
        start_time = datetime.now()
        logger.info("Generating SHAP explanation for single instance")

        # Convert instance to DataFrame
        features_df = pd.DataFrame([instance])

        # Compute SHAP values
        shap_values = self.compute_shap_values(model, features_df)

        # Get feature importance
        feature_importance = self.get_feature_importance(shap_values)

        # Generate top feature contributions
        top_features = self._generate_feature_contributions(
            shap_values, self.config.top_k_features
        )

        # Identify interaction effects
        interaction_effects = self._identify_interaction_effects(shap_values)

        # Calculate confidence based on SHAP value consistency
        confidence = self._calculate_explanation_confidence(shap_values)

        # Generate summary
        summary = self._generate_shap_summary(shap_values, top_features)

        # Generate technical detail
        technical_detail = self._generate_shap_technical_detail(shap_values, top_features)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"SHAP explanation generated in {processing_time:.1f}ms")

        return SHAPExplanation(
            explanation_id=self._generate_explanation_id(),
            explanation_type=ExplanationType.SHAP,
            summary=summary,
            technical_detail=technical_detail,
            confidence=confidence,
            confidence_level=self._get_confidence_level(confidence),
            provenance_hash=self._calculate_provenance_hash(instance),
            model_name=model.__class__.__name__,
            shap_values=shap_values,
            feature_importance=feature_importance,
            top_features=top_features,
            interaction_effects=interaction_effects,
        )

    def plot_summary(
        self,
        shap_values: SHAPValues,
    ) -> Figure:
        """
        Generate SHAP summary plot.

        Creates a beeswarm plot showing SHAP value distribution
        for all features. Returns a matplotlib-compatible figure.

        Args:
            shap_values: Computed SHAP values

        Returns:
            Figure object for the summary plot

        Note:
            Requires matplotlib. Returns None if not available.
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            # Create horizontal bar plot of SHAP values
            y_pos = np.arange(len(shap_values.feature_names))
            colors = ['red' if v < 0 else 'blue' for v in shap_values.shap_values]

            ax.barh(y_pos, shap_values.shap_values, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(shap_values.feature_names)
            ax.set_xlabel('SHAP Value (impact on prediction)')
            ax.set_title('Feature Importance (SHAP Values)')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

            plt.tight_layout()
            logger.info("SHAP summary plot generated")
            return fig

        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None

    def plot_force(
        self,
        shap_values: SHAPValues,
        instance_idx: int = 0,
    ) -> Figure:
        """
        Generate SHAP force plot for a single instance.

        Creates a force plot showing how each feature contributes
        to pushing the prediction from the base value.

        Args:
            shap_values: Computed SHAP values
            instance_idx: Index of instance to plot (default 0)

        Returns:
            Figure object for the force plot

        Note:
            Requires matplotlib. Returns None if not available.
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 3))

            # Sort features by absolute SHAP value
            abs_shap = [abs(v) for v in shap_values.shap_values]
            sorted_idx = np.argsort(abs_shap)[::-1]

            # Plot force diagram
            base = shap_values.base_value
            current = base

            positive_features = []
            negative_features = []

            for idx in sorted_idx:
                name = shap_values.feature_names[idx]
                value = shap_values.shap_values[idx]
                feat_val = shap_values.feature_values[idx]

                if value > 0:
                    positive_features.append((name, value, feat_val))
                else:
                    negative_features.append((name, value, feat_val))

            # Draw base value arrow
            ax.annotate(
                f'Base: {base:.3f}',
                xy=(0.1, 0.5),
                fontsize=10,
                ha='center'
            )

            # Draw prediction
            ax.annotate(
                f'Prediction: {shap_values.prediction:.3f}',
                xy=(0.9, 0.5),
                fontsize=10,
                ha='center'
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('SHAP Force Plot')

            plt.tight_layout()
            logger.info("SHAP force plot generated")
            return fig

        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _compute_kernel_shap(
        self,
        model: BaseModel,
        X: np.ndarray,
        background: np.ndarray,
        base_value: float,
    ) -> np.ndarray:
        """
        Compute SHAP values using KernelSHAP approximation.

        This is a simplified implementation for demonstration.
        Production systems should use the official shap library.

        Args:
            model: Model to explain
            X: Instances to explain (n_instances, n_features)
            background: Background data (n_background, n_features)
            base_value: Expected prediction value

        Returns:
            Array of SHAP values (n_instances, n_features)
        """
        n_instances, n_features = X.shape
        shap_values = np.zeros((n_instances, n_features))

        # For each instance
        for i in range(n_instances):
            instance = X[i]
            prediction = model.predict(instance.reshape(1, -1))[0]

            # Compute approximate SHAP values using permutation
            for j in range(n_features):
                # Create perturbation by replacing feature with background values
                X_plus = np.tile(instance, (len(background), 1))
                X_minus = X_plus.copy()

                # Replace feature j with background values
                X_minus[:, j] = background[:, j]

                # Compute marginal contribution
                pred_plus = np.mean(model.predict(X_plus))
                pred_minus = np.mean(model.predict(X_minus))

                shap_values[i, j] = pred_plus - pred_minus

            # Normalize to ensure additivity
            total_shap = np.sum(shap_values[i])
            expected_total = prediction - base_value

            if abs(total_shap) > 1e-10:
                shap_values[i] *= expected_total / total_shap

        return shap_values

    def _compute_interaction_values(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute SHAP interaction values."""
        # Simplified interaction computation based on covariance
        interactions = {}

        n_features = len(feature_names)
        for i in range(n_features):
            interactions[feature_names[i]] = {}
            for j in range(n_features):
                if i != j:
                    # Approximate interaction as product correlation
                    if len(shap_values) > 1:
                        interaction = np.corrcoef(
                            shap_values[:, i], shap_values[:, j]
                        )[0, 1]
                    else:
                        interaction = 0.0
                    interactions[feature_names[i]][feature_names[j]] = round(
                        float(interaction) if not np.isnan(interaction) else 0.0, 4
                    )

        return interactions

    def _generate_feature_contributions(
        self,
        shap_values: SHAPValues,
        top_k: int,
    ) -> List[FeatureContribution]:
        """Generate top feature contributions from SHAP values."""
        contributions = []

        # Sort by absolute SHAP value
        abs_shap = [abs(v) for v in shap_values.shap_values]
        sorted_idx = np.argsort(abs_shap)[::-1][:top_k]

        total_abs_shap = sum(abs_shap)

        for idx in sorted_idx:
            name = shap_values.feature_names[idx]
            value = shap_values.shap_values[idx]
            feat_val = shap_values.feature_values[idx]

            # Determine direction
            if value > 0.01:
                direction = ImpactDirection.INCREASE
            elif value < -0.01:
                direction = ImpactDirection.DECREASE
            else:
                direction = ImpactDirection.NO_CHANGE

            # Calculate contribution percentage
            contrib_pct = (abs(value) / total_abs_shap * 100) if total_abs_shap > 0 else 0

            # Generate description
            direction_text = "increases" if value > 0 else "decreases"
            description = (
                f"{name} at {feat_val:.3f} {direction_text} prediction by {abs(value):.4f}"
            )

            contributions.append(
                FeatureContribution(
                    feature_name=name,
                    feature_value=feat_val,
                    contribution=value,
                    contribution_percent=round(contrib_pct, 1),
                    direction=direction,
                    unit=None,  # Units depend on feature
                    description=description,
                )
            )

        return contributions

    def _identify_interaction_effects(
        self,
        shap_values: SHAPValues,
    ) -> List[str]:
        """Identify notable interaction effects from SHAP values."""
        effects = []

        if shap_values.interaction_values:
            for feat1, interactions in shap_values.interaction_values.items():
                for feat2, interaction in interactions.items():
                    if abs(interaction) > 0.5:
                        if interaction > 0:
                            effects.append(
                                f"Positive interaction between {feat1} and {feat2} "
                                f"(correlation: {interaction:.2f})"
                            )
                        else:
                            effects.append(
                                f"Negative interaction between {feat1} and {feat2} "
                                f"(correlation: {interaction:.2f})"
                            )

        return effects[:5]  # Limit to top 5 interactions

    def _calculate_explanation_confidence(
        self,
        shap_values: SHAPValues,
    ) -> float:
        """Calculate confidence in SHAP explanation."""
        # Base confidence on additivity check
        expected = shap_values.base_value + sum(shap_values.shap_values)
        actual = shap_values.prediction
        additivity_error = abs(expected - actual)

        # Higher error = lower confidence
        if additivity_error < 0.001:
            confidence = 0.98
        elif additivity_error < 0.01:
            confidence = 0.95
        elif additivity_error < 0.05:
            confidence = 0.85
        else:
            confidence = max(0.5, 1.0 - additivity_error)

        return round(confidence, 3)

    def _generate_shap_summary(
        self,
        shap_values: SHAPValues,
        top_features: List[FeatureContribution],
    ) -> str:
        """Generate plain language summary of SHAP explanation."""
        if not top_features:
            return "No significant feature contributions identified."

        top_feature = top_features[0]
        direction = "higher" if top_feature.contribution > 0 else "lower"

        summary = (
            f"The model predicts {shap_values.prediction:.3f} "
            f"(baseline: {shap_values.base_value:.3f}). "
            f"The most important factor is {top_feature.feature_name} "
            f"(value: {top_feature.feature_value:.2f}), "
            f"which pushes the prediction {direction}."
        )

        if len(top_features) > 1:
            other_important = [f.feature_name for f in top_features[1:3]]
            summary += f" Other important factors: {', '.join(other_important)}."

        return summary

    def _generate_shap_technical_detail(
        self,
        shap_values: SHAPValues,
        top_features: List[FeatureContribution],
    ) -> str:
        """Generate technical detail for SHAP explanation."""
        detail = []
        detail.append(f"SHAP Analysis (KernelSHAP method):\n")
        detail.append(f"Base value (E[f(X)]): {shap_values.base_value:.6f}\n")
        detail.append(f"Prediction: {shap_values.prediction:.6f}\n")
        detail.append(f"Total SHAP contribution: {sum(shap_values.shap_values):.6f}\n\n")

        detail.append("Top feature contributions:\n")
        for feat in top_features:
            detail.append(
                f"  {feat.feature_name}: {feat.contribution:+.6f} "
                f"({feat.contribution_percent:.1f}%)\n"
            )

        return "".join(detail)

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Map confidence score to confidence level."""
        if confidence > 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif confidence > 0.85:
            return ConfidenceLevel.HIGH
        elif confidence > 0.70:
            return ConfidenceLevel.MEDIUM
        elif confidence > 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _calculate_provenance_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _generate_explanation_id(self) -> str:
        """Generate unique explanation ID."""
        return f"shap-{uuid.uuid4().hex[:12]}"
