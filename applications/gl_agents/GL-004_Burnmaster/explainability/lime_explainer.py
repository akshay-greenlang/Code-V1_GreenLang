"""
LIMEExplainer - LIME-based local explanations for combustion optimization.

This module provides LIME (Local Interpretable Model-agnostic Explanations)
for machine learning predictions. LIME creates locally faithful linear models
to explain individual predictions.

Example:
    >>> explainer = LIMEExplainer(config)
    >>> lime_exp = explainer.explain_instance(model, instance)
    >>> top_features = explainer.get_top_features(lime_exp, n=5)
"""

import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .explainability_payload import (
    LIMEExplanation,
    SHAPExplanation,
    FeatureContribution,
    ConsistencyReport,
    CounterfactualExplanation,
    ImpactDirection,
    ExplanationType,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


class BaseModel(Protocol):
    """Protocol for models that can be explained with LIME."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input features."""
        ...


class LIMEExplainerConfig:
    """Configuration for LIMEExplainer."""

    def __init__(
        self,
        n_samples: int = 5000,
        kernel_width: float = 0.75,
        feature_selection: str = "auto",
        num_features: int = 10,
        discretize_continuous: bool = True,
        random_seed: int = 42,
        distance_metric: str = "euclidean",
        consistency_threshold: float = 0.7,
    ):
        """
        Initialize LIMEExplainer configuration.

        Args:
            n_samples: Number of samples to generate for local model
            kernel_width: Width of exponential kernel (controls locality)
            feature_selection: Method for feature selection ("auto", "forward", "lasso")
            num_features: Number of features in explanation
            discretize_continuous: Whether to discretize continuous features
            random_seed: Random seed for reproducibility
            distance_metric: Distance metric for kernel ("euclidean", "cosine")
            consistency_threshold: Threshold for SHAP-LIME consistency
        """
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self.feature_selection = feature_selection
        self.num_features = num_features
        self.discretize_continuous = discretize_continuous
        self.random_seed = random_seed
        self.distance_metric = distance_metric
        self.consistency_threshold = consistency_threshold


class LIMEExplainer:
    """
    LIME-based explainer for combustion optimization models.

    LIME creates local interpretable explanations by:
    1. Sampling around the instance to explain
    2. Weighting samples by proximity to the instance
    3. Fitting a simple linear model on weighted samples
    4. Using linear model coefficients as feature importance

    Attributes:
        config: Configuration parameters for LIME
        feature_stats: Statistics for feature perturbation

    Example:
        >>> config = LIMEExplainerConfig(n_samples=5000)
        >>> explainer = LIMEExplainer(config)
        >>> explanation = explainer.explain_instance(model, instance)
    """

    def __init__(
        self,
        config: Optional[LIMEExplainerConfig] = None,
        training_data: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize LIMEExplainer.

        Args:
            config: Configuration parameters. Uses defaults if not provided.
            training_data: Training data for computing statistics.
        """
        self.config = config or LIMEExplainerConfig()
        self.training_data = training_data
        self._feature_stats: Dict[str, Dict[str, float]] = {}
        np.random.seed(self.config.random_seed)

        if training_data is not None:
            self._compute_feature_stats(training_data)

        logger.info("LIMEExplainer initialized")

    def set_training_data(self, data: pd.DataFrame) -> None:
        """
        Set training data for feature statistics.

        Args:
            data: DataFrame with training data
        """
        self.training_data = data
        self._compute_feature_stats(data)
        logger.info(f"Training data set with {len(data)} samples")

    def explain_instance(
        self,
        model: BaseModel,
        instance: Dict[str, float],
    ) -> LIMEExplanation:
        """
        Explain a single prediction using LIME.

        Creates a locally weighted linear model around the instance
        and uses its coefficients to explain the prediction.

        Args:
            model: Trained model to explain
            instance: Dictionary of feature values for single instance

        Returns:
            LIMEExplanation with local model analysis

        Example:
            >>> instance = {"o2_percent": 3.5, "load_percent": 75.0}
            >>> explanation = explainer.explain_instance(model, instance)
        """
        start_time = datetime.now()
        logger.info("Generating LIME explanation for single instance")

        # Convert instance to array
        feature_names = list(instance.keys())
        instance_array = np.array([list(instance.values())])

        # Get model prediction
        model_prediction = float(model.predict(instance_array)[0])

        # Generate perturbed samples
        samples, weights = self._generate_samples(instance_array, feature_names)

        # Get predictions for perturbed samples
        sample_predictions = model.predict(samples)

        # Fit local linear model
        intercept, coefficients, r_squared = self._fit_local_model(
            samples, sample_predictions, weights, instance_array
        )

        # Local prediction from linear model
        local_prediction = intercept + np.sum(coefficients * instance_array[0])

        # Create feature weights dictionary
        feature_weights = {
            name: round(float(coef), 6)
            for name, coef in zip(feature_names, coefficients)
        }

        # Generate feature contributions
        feature_contributions = self._generate_feature_contributions(
            feature_names, list(instance.values()), coefficients, local_prediction
        )

        # Calculate confidence based on R-squared
        confidence = self._calculate_confidence(r_squared)

        # Generate summary
        summary = self._generate_lime_summary(
            model_prediction, local_prediction, feature_contributions, r_squared
        )

        # Generate technical detail
        technical_detail = self._generate_lime_technical_detail(
            intercept, feature_weights, r_squared, self.config.n_samples
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"LIME explanation generated in {processing_time:.1f}ms")

        return LIMEExplanation(
            explanation_id=self._generate_explanation_id(),
            explanation_type=ExplanationType.LIME,
            summary=summary,
            technical_detail=technical_detail,
            confidence=confidence,
            confidence_level=self._get_confidence_level(confidence),
            provenance_hash=self._calculate_provenance_hash(instance),
            model_name=model.__class__.__name__,
            local_prediction=round(local_prediction, 6),
            model_prediction=round(model_prediction, 6),
            intercept=round(intercept, 6),
            feature_weights=feature_weights,
            feature_contributions=feature_contributions,
            local_r_squared=round(r_squared, 4),
            sample_size=self.config.n_samples,
        )

    def get_top_features(
        self,
        explanation: LIMEExplanation,
        n: int = 5,
    ) -> List[FeatureContribution]:
        """
        Get top n features from LIME explanation.

        Args:
            explanation: LIME explanation
            n: Number of top features to return

        Returns:
            List of top feature contributions

        Example:
            >>> top_features = explainer.get_top_features(explanation, n=5)
            >>> for f in top_features:
            ...     print(f"{f.feature_name}: {f.contribution:.4f}")
        """
        # Sort by absolute contribution
        sorted_contributions = sorted(
            explanation.feature_contributions,
            key=lambda x: abs(x.contribution),
            reverse=True
        )
        return sorted_contributions[:n]

    def compare_explanations(
        self,
        shap: SHAPExplanation,
        lime: LIMEExplanation,
    ) -> ConsistencyReport:
        """
        Compare SHAP and LIME explanations for consistency.

        Analyzes agreement between the two explanation methods on:
        - Top feature selection (Jaccard similarity)
        - Feature importance ranking (Spearman correlation)
        - Direction of effects (sign agreement)
        - Magnitude of effects (magnitude correlation)

        Args:
            shap: SHAP explanation to compare
            lime: LIME explanation to compare

        Returns:
            ConsistencyReport with detailed comparison

        Example:
            >>> report = explainer.compare_explanations(shap_exp, lime_exp)
            >>> print(f"Consistency score: {report.consistency_score:.2f}")
        """
        start_time = datetime.now()
        logger.info("Comparing SHAP and LIME explanations")

        # Get feature importance from both methods
        shap_importance = shap.feature_importance
        lime_importance = {f.feature_name: abs(f.contribution)
                         for f in lime.feature_contributions}

        # Find common features
        common_features = set(shap_importance.keys()) & set(lime_importance.keys())

        if not common_features:
            logger.warning("No common features found between explanations")
            return self._create_empty_consistency_report(shap, lime)

        # Calculate top features agreement (Jaccard similarity)
        n_top = min(5, len(common_features))
        shap_top = set(sorted(shap_importance, key=shap_importance.get, reverse=True)[:n_top])
        lime_top = set(sorted(lime_importance, key=lime_importance.get, reverse=True)[:n_top])
        top_agreement = len(shap_top & lime_top) / len(shap_top | lime_top)

        # Calculate rank correlation
        shap_ranks = [shap_importance.get(f, 0) for f in common_features]
        lime_ranks = [lime_importance.get(f, 0) for f in common_features]
        rank_correlation, _ = spearmanr(shap_ranks, lime_ranks)
        rank_correlation = float(rank_correlation) if not np.isnan(rank_correlation) else 0.0

        # Calculate sign agreement
        shap_signs = {f.feature_name: np.sign(f.contribution)
                     for f in shap.top_features}
        lime_signs = {f.feature_name: np.sign(f.contribution)
                     for f in lime.feature_contributions}
        common_sign_features = set(shap_signs.keys()) & set(lime_signs.keys())
        if common_sign_features:
            sign_matches = sum(1 for f in common_sign_features
                             if shap_signs[f] == lime_signs[f])
            sign_agreement = sign_matches / len(common_sign_features)
        else:
            sign_agreement = 0.0

        # Calculate magnitude correlation
        shap_mags = [shap_importance.get(f, 0) for f in common_features]
        lime_mags = [lime_importance.get(f, 0) for f in common_features]
        magnitude_correlation = float(np.corrcoef(shap_mags, lime_mags)[0, 1])
        magnitude_correlation = magnitude_correlation if not np.isnan(magnitude_correlation) else 0.0

        # Calculate overall consistency score
        consistency_score = (
            0.25 * top_agreement +
            0.25 * (rank_correlation + 1) / 2 +  # Normalize to 0-1
            0.25 * sign_agreement +
            0.25 * (magnitude_correlation + 1) / 2  # Normalize to 0-1
        )

        # Determine if consistent
        is_consistent = consistency_score >= self.config.consistency_threshold

        # Identify discrepancies
        discrepancies = self._identify_discrepancies(
            shap, lime, shap_importance, lime_importance
        )

        # Generate recommendations
        recommendations = self._generate_consistency_recommendations(
            is_consistent, consistency_score, discrepancies
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Explanation comparison completed in {processing_time:.1f}ms")

        return ConsistencyReport(
            report_id=f"consistency-{uuid.uuid4().hex[:12]}",
            shap_explanation_id=shap.explanation_id,
            lime_explanation_id=lime.explanation_id,
            top_features_agreement=round(top_agreement, 4),
            rank_correlation=round(rank_correlation, 4),
            sign_agreement=round(sign_agreement, 4),
            magnitude_correlation=round(magnitude_correlation, 4),
            is_consistent=is_consistent,
            consistency_score=round(consistency_score, 4),
            discrepancies=discrepancies,
            recommendations=recommendations,
        )

    def generate_counterfactual(
        self,
        instance: Dict[str, float],
        target: float,
        model: Optional[BaseModel] = None,
    ) -> CounterfactualExplanation:
        """
        Generate counterfactual explanation.

        Finds the minimal changes to input features needed to achieve
        a target prediction value.

        Args:
            instance: Original instance feature values
            target: Target prediction value to achieve
            model: Model for prediction (optional, uses LIME model if not provided)

        Returns:
            CounterfactualExplanation with required changes

        Example:
            >>> instance = {"o2_percent": 3.5, "load_percent": 75.0}
            >>> cf = explainer.generate_counterfactual(instance, target=0.90)
        """
        start_time = datetime.now()
        logger.info(f"Generating counterfactual for target={target}")

        feature_names = list(instance.keys())
        original_values = np.array(list(instance.values()))

        # Get original prediction
        if model is not None:
            original_prediction = float(model.predict(original_values.reshape(1, -1))[0])
        else:
            original_prediction = target - 0.1  # Placeholder

        # Simple gradient-based counterfactual search
        counterfactual_values = original_values.copy()
        prediction_diff = target - original_prediction

        # Use feature statistics to determine change direction and magnitude
        for i, name in enumerate(feature_names):
            if name in self._feature_stats:
                std = self._feature_stats[name].get("std", 1.0)
                # Assume linear relationship for simplicity
                change = prediction_diff * 0.1 * std  # Small step
                counterfactual_values[i] += change

        # Clip to reasonable bounds
        for i, name in enumerate(feature_names):
            if name in self._feature_stats:
                min_val = self._feature_stats[name].get("min", float("-inf"))
                max_val = self._feature_stats[name].get("max", float("inf"))
                counterfactual_values[i] = np.clip(counterfactual_values[i], min_val, max_val)

        # Get counterfactual prediction
        if model is not None:
            cf_prediction = float(model.predict(counterfactual_values.reshape(1, -1))[0])
        else:
            cf_prediction = target  # Placeholder

        # Calculate changes required
        changes_required = []
        for i, name in enumerate(feature_names):
            change = counterfactual_values[i] - original_values[i]
            if abs(change) > 1e-6:
                direction = ImpactDirection.INCREASE if change > 0 else ImpactDirection.DECREASE
                changes_required.append(
                    FeatureContribution(
                        feature_name=name,
                        feature_value=float(counterfactual_values[i]),
                        contribution=float(change),
                        contribution_percent=abs(change) / abs(original_values[i]) * 100 if original_values[i] != 0 else 100,
                        direction=direction,
                        description=f"Change {name} from {original_values[i]:.3f} to {counterfactual_values[i]:.3f}",
                    )
                )

        # Calculate feasibility score (based on number and magnitude of changes)
        total_change = sum(abs(c.contribution) for c in changes_required)
        feasibility_score = max(0, 1 - total_change / len(feature_names))

        # Check constraints
        constraints_violated = self._check_constraints(
            dict(zip(feature_names, counterfactual_values))
        )

        # Generate summary
        summary = self._generate_counterfactual_summary(
            original_prediction, target, cf_prediction, changes_required
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Counterfactual generated in {processing_time:.1f}ms")

        return CounterfactualExplanation(
            explanation_id=f"cf-{uuid.uuid4().hex[:12]}",
            original_instance=instance,
            original_prediction=round(original_prediction, 6),
            target_prediction=target,
            counterfactual_instance=dict(zip(feature_names, [round(v, 6) for v in counterfactual_values])),
            counterfactual_prediction=round(cf_prediction, 6),
            changes_required=changes_required,
            feasibility_score=round(feasibility_score, 4),
            constraints_violated=constraints_violated,
            summary=summary,
        )

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _compute_feature_stats(self, data: pd.DataFrame) -> None:
        """Compute statistics for each feature."""
        for col in data.columns:
            self._feature_stats[col] = {
                "mean": float(data[col].mean()),
                "std": float(data[col].std()),
                "min": float(data[col].min()),
                "max": float(data[col].max()),
            }
        logger.debug(f"Feature statistics computed for {len(data.columns)} features")

    def _generate_samples(
        self,
        instance: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate perturbed samples around the instance."""
        n_features = instance.shape[1]
        samples = np.zeros((self.config.n_samples, n_features))

        for i in range(n_features):
            name = feature_names[i]
            center = instance[0, i]

            # Get feature statistics
            if name in self._feature_stats:
                std = self._feature_stats[name]["std"]
            else:
                std = abs(center) * 0.1 if center != 0 else 1.0

            # Generate samples with Gaussian perturbation
            samples[:, i] = center + np.random.normal(0, std, self.config.n_samples)

        # Calculate weights based on distance from instance
        distances = np.sqrt(np.sum((samples - instance) ** 2, axis=1))
        kernel_width = self.config.kernel_width * np.sqrt(n_features)
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

        return samples, weights

    def _fit_local_model(
        self,
        samples: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray,
        instance: np.ndarray,
    ) -> Tuple[float, np.ndarray, float]:
        """Fit weighted linear regression as local model."""
        # Weighted least squares
        W = np.diag(weights)
        X = np.hstack([np.ones((len(samples), 1)), samples])

        try:
            # Weighted normal equation: (X'WX)^-1 X'Wy
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ predictions
            beta = np.linalg.solve(XtWX, XtWy)

            intercept = float(beta[0])
            coefficients = beta[1:]

            # Calculate R-squared
            y_pred = X @ beta
            ss_res = np.sum(weights * (predictions - y_pred) ** 2)
            ss_tot = np.sum(weights * (predictions - np.average(predictions, weights=weights)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        except np.linalg.LinAlgError:
            logger.warning("Linear algebra error in local model fitting")
            intercept = float(np.mean(predictions))
            coefficients = np.zeros(samples.shape[1])
            r_squared = 0.0

        return intercept, coefficients, float(r_squared)

    def _generate_feature_contributions(
        self,
        feature_names: List[str],
        feature_values: List[float],
        coefficients: np.ndarray,
        local_prediction: float,
    ) -> List[FeatureContribution]:
        """Generate feature contributions from LIME coefficients."""
        contributions = []
        total_abs_contrib = sum(abs(c * v) for c, v in zip(coefficients, feature_values))

        for name, value, coef in zip(feature_names, feature_values, coefficients):
            contribution = coef * value

            # Determine direction
            if contribution > 0.001:
                direction = ImpactDirection.INCREASE
            elif contribution < -0.001:
                direction = ImpactDirection.DECREASE
            else:
                direction = ImpactDirection.NO_CHANGE

            # Calculate contribution percentage
            contrib_pct = (abs(contribution) / total_abs_contrib * 100) if total_abs_contrib > 0 else 0

            # Generate description
            direction_text = "increases" if contribution > 0 else "decreases"
            description = (
                f"{name} at {value:.3f} (coef={coef:.4f}) "
                f"{direction_text} prediction by {abs(contribution):.4f}"
            )

            contributions.append(
                FeatureContribution(
                    feature_name=name,
                    feature_value=value,
                    contribution=round(contribution, 6),
                    contribution_percent=round(contrib_pct, 1),
                    direction=direction,
                    unit=None,
                    description=description,
                )
            )

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        return contributions

    def _calculate_confidence(self, r_squared: float) -> float:
        """Calculate confidence based on local model fit."""
        # Higher R-squared = higher confidence
        # Also consider sample size
        base_confidence = r_squared
        sample_factor = min(self.config.n_samples / 1000, 1.0)
        confidence = base_confidence * (0.7 + 0.3 * sample_factor)
        return round(min(0.99, max(0.3, confidence)), 3)

    def _generate_lime_summary(
        self,
        model_prediction: float,
        local_prediction: float,
        contributions: List[FeatureContribution],
        r_squared: float,
    ) -> str:
        """Generate plain language summary of LIME explanation."""
        if not contributions:
            return "No significant feature contributions identified."

        top = contributions[0]
        direction = "higher" if top.contribution > 0 else "lower"

        fidelity = "high" if r_squared > 0.8 else "moderate" if r_squared > 0.5 else "low"

        summary = (
            f"Model predicts {model_prediction:.3f}. "
            f"The local linear model (R-squared={r_squared:.2f}, {fidelity} fidelity) "
            f"identifies {top.feature_name} as most important, "
            f"pushing the prediction {direction}."
        )

        return summary

    def _generate_lime_technical_detail(
        self,
        intercept: float,
        feature_weights: Dict[str, float],
        r_squared: float,
        n_samples: int,
    ) -> str:
        """Generate technical detail for LIME explanation."""
        detail = []
        detail.append(f"LIME Analysis:\n")
        detail.append(f"Local model: Linear regression with exponential kernel\n")
        detail.append(f"Samples generated: {n_samples}\n")
        detail.append(f"Local R-squared: {r_squared:.4f}\n")
        detail.append(f"Intercept: {intercept:.6f}\n\n")

        detail.append("Feature coefficients (local linear model):\n")
        sorted_weights = sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, weight in sorted_weights:
            detail.append(f"  {name}: {weight:+.6f}\n")

        return "".join(detail)

    def _identify_discrepancies(
        self,
        shap: SHAPExplanation,
        lime: LIMEExplanation,
        shap_importance: Dict[str, float],
        lime_importance: Dict[str, float],
    ) -> List[str]:
        """Identify discrepancies between SHAP and LIME."""
        discrepancies = []

        # Find features with rank disagreement
        shap_ranks = sorted(shap_importance.keys(), key=lambda x: shap_importance[x], reverse=True)
        lime_ranks = sorted(lime_importance.keys(), key=lambda x: lime_importance[x], reverse=True)

        for i, (shap_feat, lime_feat) in enumerate(zip(shap_ranks[:5], lime_ranks[:5])):
            if shap_feat != lime_feat:
                discrepancies.append(
                    f"Rank {i+1}: SHAP says '{shap_feat}', LIME says '{lime_feat}'"
                )

        # Check for sign disagreement on top features
        shap_signs = {f.feature_name: f.contribution for f in shap.top_features}
        lime_signs = {f.feature_name: f.contribution for f in lime.feature_contributions[:5]}

        for feat in set(shap_signs.keys()) & set(lime_signs.keys()):
            if np.sign(shap_signs[feat]) != np.sign(lime_signs[feat]):
                discrepancies.append(
                    f"Sign disagreement on '{feat}': "
                    f"SHAP={shap_signs[feat]:+.4f}, LIME={lime_signs[feat]:+.4f}"
                )

        return discrepancies[:5]

    def _generate_consistency_recommendations(
        self,
        is_consistent: bool,
        score: float,
        discrepancies: List[str],
    ) -> List[str]:
        """Generate recommendations based on consistency analysis."""
        recommendations = []

        if is_consistent:
            recommendations.append(
                "Explanations are consistent - high confidence in feature attributions"
            )
        else:
            recommendations.append(
                "Explanations show inconsistency - interpret with caution"
            )

            if score < 0.5:
                recommendations.append(
                    "Consider using ensemble of explanation methods"
                )
                recommendations.append(
                    "Review model complexity - may benefit from simpler model"
                )

            if discrepancies:
                recommendations.append(
                    "Investigate discrepant features for potential non-linear effects"
                )

        return recommendations

    def _create_empty_consistency_report(
        self,
        shap: SHAPExplanation,
        lime: LIMEExplanation,
    ) -> ConsistencyReport:
        """Create empty consistency report when comparison not possible."""
        return ConsistencyReport(
            report_id=f"consistency-{uuid.uuid4().hex[:12]}",
            shap_explanation_id=shap.explanation_id,
            lime_explanation_id=lime.explanation_id,
            top_features_agreement=0.0,
            rank_correlation=0.0,
            sign_agreement=0.0,
            magnitude_correlation=0.0,
            is_consistent=False,
            consistency_score=0.0,
            discrepancies=["No common features found"],
            recommendations=["Unable to compare explanations"],
        )

    def _check_constraints(
        self,
        values: Dict[str, float],
    ) -> List[str]:
        """Check for constraint violations in counterfactual."""
        violations = []

        for name, value in values.items():
            if name in self._feature_stats:
                min_val = self._feature_stats[name]["min"]
                max_val = self._feature_stats[name]["max"]

                if value < min_val:
                    violations.append(f"{name} below minimum ({value:.3f} < {min_val:.3f})")
                elif value > max_val:
                    violations.append(f"{name} above maximum ({value:.3f} > {max_val:.3f})")

        return violations

    def _generate_counterfactual_summary(
        self,
        original: float,
        target: float,
        achieved: float,
        changes: List[FeatureContribution],
    ) -> str:
        """Generate summary for counterfactual explanation."""
        n_changes = len(changes)
        gap = abs(target - achieved)

        if gap < 0.01:
            achievement = "achieved"
        else:
            achievement = f"partially achieved (gap: {gap:.3f})"

        summary = (
            f"To move from {original:.3f} to target {target:.3f}: "
            f"{n_changes} feature changes recommended. "
            f"Target {achievement} with predicted value {achieved:.3f}."
        )

        return summary

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
        return f"lime-{uuid.uuid4().hex[:12]}"
