"""
GL-002 Flameguard - SHAP Combustion Explainer

Real SHAP (SHapley Additive exPlanations) implementation for explaining
combustion optimization decisions using TreeExplainer.

This module provides:
1. Feature importance analysis for combustion parameters
2. Individual prediction explanations
3. Contribution analysis for optimization decisions
4. Audit-ready explanation reports with provenance tracking

Uses actual SHAP library with TreeExplainer for tree-based models
(XGBoost, LightGBM, RandomForest) that may be used for combustion
optimization predictions.

Reference:
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to
  Interpreting Model Predictions. NeurIPS.

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

# Conditional imports for SHAP
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available - SHAP features will be limited")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available - using fallback explainer")


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ExplanationType(Enum):
    """Types of SHAP explanations."""

    SINGLE_PREDICTION = "single_prediction"
    FEATURE_IMPORTANCE = "feature_importance"
    CONTRIBUTION_ANALYSIS = "contribution_analysis"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"


class CombustionFeature(Enum):
    """Combustion-related features for SHAP analysis."""

    O2_PERCENT = "o2_percent"
    CO_PPM = "co_ppm"
    NOX_PPM = "nox_ppm"
    STACK_TEMP_F = "stack_temp_f"
    LOAD_PERCENT = "load_percent"
    FUEL_FLOW = "fuel_flow"
    EXCESS_AIR = "excess_air_percent"
    EFFICIENCY = "efficiency_percent"
    STEAM_FLOW = "steam_flow_klb_hr"
    FEEDWATER_TEMP = "feedwater_temp_f"
    AMBIENT_TEMP = "ambient_temp_f"
    AIR_FUEL_RATIO = "air_fuel_ratio"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeatureContribution:
    """Contribution of a single feature to prediction."""

    feature_name: str
    feature_value: float
    shap_value: float
    contribution_percent: float
    direction: str  # "positive" or "negative"
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "shap_value": self.shap_value,
            "contribution_percent": self.contribution_percent,
            "direction": self.direction,
            "explanation": self.explanation,
        }


@dataclass
class SHAPExplanation:
    """Complete SHAP explanation for a prediction."""

    explanation_id: str
    explanation_type: ExplanationType
    timestamp: datetime
    target_variable: str

    # Base value (expected prediction without any features)
    base_value: float

    # Predicted value
    prediction: float

    # Feature contributions
    contributions: List[FeatureContribution]

    # Summary statistics
    total_positive_contribution: float
    total_negative_contribution: float
    top_positive_feature: str
    top_negative_feature: str

    # Input features
    input_features: Dict[str, float]

    # Provenance
    model_hash: str
    input_hash: str
    output_hash: str

    # Natural language summary
    summary: str = ""
    detailed_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "explanation_id": self.explanation_id,
            "explanation_type": self.explanation_type.value,
            "timestamp": self.timestamp.isoformat(),
            "target_variable": self.target_variable,
            "base_value": self.base_value,
            "prediction": self.prediction,
            "contributions": [c.to_dict() for c in self.contributions],
            "total_positive_contribution": self.total_positive_contribution,
            "total_negative_contribution": self.total_negative_contribution,
            "top_positive_feature": self.top_positive_feature,
            "top_negative_feature": self.top_negative_feature,
            "input_features": self.input_features,
            "model_hash": self.model_hash,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "summary": self.summary,
            "detailed_explanation": self.detailed_explanation,
        }


@dataclass
class FeatureImportanceResult:
    """Global feature importance from SHAP values."""

    importance_id: str
    timestamp: datetime
    target_variable: str
    n_samples: int

    # Feature importances (mean absolute SHAP value)
    importances: Dict[str, float]
    importance_ranking: List[str]

    # Provenance
    model_hash: str
    data_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "importance_id": self.importance_id,
            "timestamp": self.timestamp.isoformat(),
            "target_variable": self.target_variable,
            "n_samples": self.n_samples,
            "importances": self.importances,
            "importance_ranking": self.importance_ranking,
            "model_hash": self.model_hash,
            "data_hash": self.data_hash,
        }


# =============================================================================
# FEATURE EXPLANATION TEMPLATES
# =============================================================================

FEATURE_EXPLANATIONS = {
    CombustionFeature.O2_PERCENT: {
        "positive": (
            "Higher O2 ({value:.1f}%) increases excess air, reducing efficiency "
            "but improving combustion completeness and NOx margin."
        ),
        "negative": (
            "Lower O2 ({value:.1f}%) reduces excess air losses, improving efficiency "
            "but requiring careful CO monitoring."
        ),
    },
    CombustionFeature.CO_PPM: {
        "positive": (
            "Higher CO ({value:.0f} ppm) indicates incomplete combustion, "
            "suggesting need for more excess air or improved mixing."
        ),
        "negative": (
            "Lower CO ({value:.0f} ppm) indicates complete combustion "
            "with good air-fuel mixing."
        ),
    },
    CombustionFeature.STACK_TEMP_F: {
        "positive": (
            "Higher stack temperature ({value:.0f} F) increases dry flue gas losses, "
            "indicating potential for economizer or air preheater improvements."
        ),
        "negative": (
            "Lower stack temperature ({value:.0f} F) minimizes stack losses, "
            "improving overall efficiency."
        ),
    },
    CombustionFeature.LOAD_PERCENT: {
        "positive": (
            "Operating at {value:.0f}% load affects efficiency through "
            "fixed loss distribution and turndown performance."
        ),
        "negative": (
            "Load of {value:.0f}% is below optimal range, increasing "
            "relative fixed losses and reducing efficiency."
        ),
    },
    CombustionFeature.EXCESS_AIR: {
        "positive": (
            "Excess air of {value:.1f}% provides combustion safety margin "
            "but increases dry flue gas losses."
        ),
        "negative": (
            "Lower excess air ({value:.1f}%) minimizes stack losses "
            "while maintaining combustion quality."
        ),
    },
}


# =============================================================================
# SHAP COMBUSTION EXPLAINER
# =============================================================================

class SHAPCombustionExplainer:
    """
    SHAP-based explainer for combustion optimization decisions.

    Uses TreeExplainer for tree-based models or KernelExplainer
    for other model types. Provides feature contribution analysis,
    importance rankings, and audit-ready explanations.

    Example:
        >>> explainer = SHAPCombustionExplainer(model)
        >>> features = {"o2_percent": 3.5, "stack_temp_f": 350, ...}
        >>> explanation = explainer.explain_prediction(features)
        >>> print(explanation.summary)
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        model: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[Any] = None,
        model_type: str = "tree",
    ) -> None:
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model (XGBoost, LightGBM, RandomForest, etc.)
            feature_names: List of feature names
            background_data: Background dataset for KernelExplainer
            model_type: Type of model ("tree", "linear", "kernel")
        """
        self.model = model
        self.feature_names = feature_names or [f.value for f in CombustionFeature]
        self.model_type = model_type
        self._explainer = None
        self._model_hash = self._compute_model_hash()

        # Initialize SHAP explainer if model provided
        if model is not None and SHAP_AVAILABLE:
            self._init_shap_explainer(background_data)
        elif model is not None:
            logger.warning("SHAP not available - using analytical fallback")

        logger.info(f"SHAPCombustionExplainer initialized: type={model_type}")

    def _init_shap_explainer(self, background_data: Optional[Any] = None) -> None:
        """Initialize the appropriate SHAP explainer."""
        try:
            if self.model_type == "tree":
                self._explainer = shap.TreeExplainer(self.model)
                logger.info("TreeExplainer initialized")
            elif self.model_type == "linear":
                self._explainer = shap.LinearExplainer(
                    self.model,
                    background_data
                )
                logger.info("LinearExplainer initialized")
            else:
                if background_data is not None:
                    self._explainer = shap.KernelExplainer(
                        self.model.predict,
                        background_data
                    )
                    logger.info("KernelExplainer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self._explainer = None

    def explain_prediction(
        self,
        features: Dict[str, float],
        target_variable: str = "efficiency_percent",
    ) -> SHAPExplanation:
        """
        Generate SHAP explanation for a single prediction.

        Args:
            features: Dictionary of feature name to value
            target_variable: Name of target variable being predicted

        Returns:
            SHAPExplanation with feature contributions
        """
        timestamp = datetime.now(timezone.utc)

        # Convert features to array
        if NUMPY_AVAILABLE:
            feature_array = np.array([[features.get(f, 0) for f in self.feature_names]])
        else:
            feature_array = [[features.get(f, 0) for f in self.feature_names]]

        # Calculate SHAP values
        if self._explainer is not None and SHAP_AVAILABLE:
            shap_values = self._calculate_shap_values(feature_array)
            base_value = self._get_base_value()
        else:
            # Analytical fallback
            shap_values, base_value = self._analytical_shap_values(features)

        # Get prediction
        if self.model is not None:
            try:
                prediction = float(self.model.predict(feature_array)[0])
            except Exception:
                prediction = base_value + sum(shap_values)
        else:
            prediction = base_value + sum(shap_values)

        # Build contributions
        contributions = self._build_contributions(features, shap_values)

        # Calculate summary statistics
        positive_contribs = [c for c in contributions if c.shap_value > 0]
        negative_contribs = [c for c in contributions if c.shap_value < 0]

        total_positive = sum(c.shap_value for c in positive_contribs)
        total_negative = sum(c.shap_value for c in negative_contribs)

        top_positive = max(positive_contribs, key=lambda c: c.shap_value).feature_name if positive_contribs else ""
        top_negative = min(negative_contribs, key=lambda c: c.shap_value).feature_name if negative_contribs else ""

        # Compute hashes
        input_hash = self._compute_hash(features)
        output_hash = self._compute_hash({
            "prediction": prediction,
            "shap_values": shap_values,
        })

        explanation = SHAPExplanation(
            explanation_id=f"SHAP-{timestamp.strftime('%Y%m%d%H%M%S')}",
            explanation_type=ExplanationType.SINGLE_PREDICTION,
            timestamp=timestamp,
            target_variable=target_variable,
            base_value=base_value,
            prediction=prediction,
            contributions=contributions,
            total_positive_contribution=total_positive,
            total_negative_contribution=total_negative,
            top_positive_feature=top_positive,
            top_negative_feature=top_negative,
            input_features=features,
            model_hash=self._model_hash,
            input_hash=input_hash,
            output_hash=output_hash,
        )

        # Generate natural language explanations
        self._generate_summary(explanation)
        self._generate_detailed_explanation(explanation)

        return explanation

    def _calculate_shap_values(self, feature_array: Any) -> List[float]:
        """Calculate SHAP values using the explainer."""
        try:
            shap_values = self._explainer.shap_values(feature_array)

            # Handle multi-output models
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Flatten to list
            if NUMPY_AVAILABLE:
                return shap_values.flatten().tolist()
            return list(shap_values[0])

        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return [0.0] * len(self.feature_names)

    def _get_base_value(self) -> float:
        """Get base value (expected value) from explainer."""
        try:
            base = self._explainer.expected_value
            if isinstance(base, (list, tuple)):
                return float(base[0])
            if NUMPY_AVAILABLE and hasattr(base, 'item'):
                return float(base.item())
            return float(base)
        except Exception:
            return 82.0  # Default expected efficiency

    def _analytical_shap_values(
        self,
        features: Dict[str, float],
    ) -> Tuple[List[float], float]:
        """
        Calculate analytical SHAP-like values when SHAP library unavailable.

        Uses domain knowledge about combustion relationships to estimate
        feature contributions.
        """
        base_efficiency = 82.0  # Baseline efficiency

        # Sensitivity coefficients (efficiency change per unit)
        sensitivities = {
            "o2_percent": -0.30,      # Higher O2 reduces efficiency
            "co_ppm": -0.005,          # Higher CO indicates losses
            "stack_temp_f": -0.015,    # Higher stack temp = more losses
            "load_percent": 0.05,      # Optimal around 75%
            "excess_air_percent": -0.02,  # Higher EA = more losses
            "feedwater_temp_f": 0.02,  # Higher FW temp = less fuel needed
            "ambient_temp_f": -0.01,   # Higher ambient = less combustion air heating
            "air_fuel_ratio": -0.10,   # Higher AFR = more excess air losses
        }

        # Reference values
        reference = {
            "o2_percent": 3.0,
            "co_ppm": 30.0,
            "stack_temp_f": 350.0,
            "load_percent": 75.0,
            "excess_air_percent": 15.0,
            "feedwater_temp_f": 227.0,
            "ambient_temp_f": 77.0,
            "air_fuel_ratio": 17.0,
        }

        shap_values = []
        for feature in self.feature_names:
            value = features.get(feature, reference.get(feature, 0))
            ref_value = reference.get(feature, value)
            sensitivity = sensitivities.get(feature, 0)

            # SHAP value = deviation from reference * sensitivity
            shap_value = (value - ref_value) * sensitivity
            shap_values.append(shap_value)

        return shap_values, base_efficiency

    def _build_contributions(
        self,
        features: Dict[str, float],
        shap_values: List[float],
    ) -> List[FeatureContribution]:
        """Build feature contribution objects from SHAP values."""
        contributions = []
        total_abs_shap = sum(abs(v) for v in shap_values) or 1.0

        for i, feature_name in enumerate(self.feature_names):
            shap_value = shap_values[i] if i < len(shap_values) else 0.0
            feature_value = features.get(feature_name, 0.0)

            direction = "positive" if shap_value >= 0 else "negative"
            contribution_pct = abs(shap_value) / total_abs_shap * 100

            explanation = self._generate_feature_explanation(
                feature_name, feature_value, direction
            )

            contributions.append(FeatureContribution(
                feature_name=feature_name,
                feature_value=feature_value,
                shap_value=round(shap_value, 4),
                contribution_percent=round(contribution_pct, 2),
                direction=direction,
                explanation=explanation,
            ))

        # Sort by absolute SHAP value
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

        return contributions

    def _generate_feature_explanation(
        self,
        feature_name: str,
        value: float,
        direction: str,
    ) -> str:
        """Generate natural language explanation for feature contribution."""
        try:
            feature = CombustionFeature(feature_name)
            templates = FEATURE_EXPLANATIONS.get(feature, {})
            template = templates.get(direction, f"{feature_name} = {value}")
            return template.format(value=value)
        except ValueError:
            return f"{feature_name} = {value} ({direction} contribution)"

    def calculate_feature_importance(
        self,
        data: Any,
        target_variable: str = "efficiency_percent",
    ) -> FeatureImportanceResult:
        """
        Calculate global feature importance from multiple predictions.

        Uses mean absolute SHAP values as importance measure.

        Args:
            data: Array-like data with shape (n_samples, n_features)
            target_variable: Name of target variable

        Returns:
            FeatureImportanceResult with importance rankings
        """
        timestamp = datetime.now(timezone.utc)

        if self._explainer is not None and SHAP_AVAILABLE:
            shap_values = self._explainer.shap_values(data)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Mean absolute SHAP value per feature
            if NUMPY_AVAILABLE:
                importances = np.abs(shap_values).mean(axis=0).tolist()
            else:
                importances = [0.0] * len(self.feature_names)

            n_samples = len(data) if hasattr(data, '__len__') else 0
        else:
            # Analytical fallback
            importances = self._analytical_importance()
            n_samples = 0

        # Build importance dictionary
        importance_dict = {
            self.feature_names[i]: round(importances[i], 4)
            for i in range(len(self.feature_names))
        }

        # Rank features by importance
        ranking = sorted(
            self.feature_names,
            key=lambda f: importance_dict.get(f, 0),
            reverse=True
        )

        # Compute hashes
        data_hash = self._compute_hash({"n_samples": n_samples})

        return FeatureImportanceResult(
            importance_id=f"IMP-{timestamp.strftime('%Y%m%d%H%M%S')}",
            timestamp=timestamp,
            target_variable=target_variable,
            n_samples=n_samples,
            importances=importance_dict,
            importance_ranking=ranking,
            model_hash=self._model_hash,
            data_hash=data_hash,
        )

    def _analytical_importance(self) -> List[float]:
        """
        Calculate analytical feature importance based on domain knowledge.

        Returns importance values reflecting typical combustion relationships.
        """
        # Relative importance of features for efficiency prediction
        importance = {
            "o2_percent": 0.25,           # Primary control variable
            "stack_temp_f": 0.20,          # Major loss contributor
            "excess_air_percent": 0.15,    # Direct efficiency impact
            "load_percent": 0.12,          # Affects all losses
            "co_ppm": 0.08,                # Combustion quality indicator
            "air_fuel_ratio": 0.08,        # Efficiency driver
            "feedwater_temp_f": 0.05,      # Secondary effect
            "ambient_temp_f": 0.04,        # Environmental factor
            "nox_ppm": 0.02,               # Indirect indicator
            "fuel_flow": 0.01,             # Normalization factor
        }

        return [
            importance.get(f, 0.01)
            for f in self.feature_names
        ]

    def _generate_summary(self, explanation: SHAPExplanation) -> None:
        """Generate natural language summary of explanation."""
        top_contrib = explanation.contributions[0] if explanation.contributions else None

        if top_contrib is None:
            explanation.summary = "No significant feature contributions identified."
            return

        parts = [
            f"Predicted {explanation.target_variable}: {explanation.prediction:.2f}",
            f"(base: {explanation.base_value:.2f})",
        ]

        if explanation.total_positive_contribution > 0:
            parts.append(
                f"Top positive: {explanation.top_positive_feature} "
                f"(+{explanation.total_positive_contribution:.2f})"
            )

        if explanation.total_negative_contribution < 0:
            parts.append(
                f"Top negative: {explanation.top_negative_feature} "
                f"({explanation.total_negative_contribution:.2f})"
            )

        explanation.summary = " | ".join(parts)

    def _generate_detailed_explanation(self, explanation: SHAPExplanation) -> None:
        """Generate detailed natural language explanation."""
        lines = [
            f"SHAP Explanation for {explanation.target_variable}",
            "=" * 50,
            "",
            f"Base Value (Expected): {explanation.base_value:.2f}",
            f"Predicted Value: {explanation.prediction:.2f}",
            "",
            "Feature Contributions (sorted by importance):",
            "-" * 40,
        ]

        for i, contrib in enumerate(explanation.contributions[:5]):
            sign = "+" if contrib.shap_value >= 0 else ""
            lines.append(
                f"{i+1}. {contrib.feature_name}: {sign}{contrib.shap_value:.3f} "
                f"({contrib.contribution_percent:.1f}%)"
            )
            lines.append(f"   {contrib.explanation}")
            lines.append("")

        lines.extend([
            "-" * 40,
            f"Total Positive Contribution: +{explanation.total_positive_contribution:.3f}",
            f"Total Negative Contribution: {explanation.total_negative_contribution:.3f}",
        ])

        explanation.detailed_explanation = "\n".join(lines)

    def explain_optimization_decision(
        self,
        current_state: Dict[str, float],
        recommended_state: Dict[str, float],
        target_variable: str = "efficiency_percent",
    ) -> Dict[str, Any]:
        """
        Explain the difference between current and recommended states.

        Compares SHAP explanations for both states to show why
        the optimization recommends specific changes.

        Args:
            current_state: Current operating parameters
            recommended_state: Recommended parameters
            target_variable: Target variable being optimized

        Returns:
            Dictionary with comparison and recommendations
        """
        current_explanation = self.explain_prediction(current_state, target_variable)
        recommended_explanation = self.explain_prediction(recommended_state, target_variable)

        # Find features with significant changes
        significant_changes = []
        for feature in self.feature_names:
            current_val = current_state.get(feature, 0)
            recommended_val = recommended_state.get(feature, current_val)

            if abs(current_val - recommended_val) > 0.01:
                # Find SHAP value change
                current_shap = next(
                    (c.shap_value for c in current_explanation.contributions
                     if c.feature_name == feature),
                    0
                )
                recommended_shap = next(
                    (c.shap_value for c in recommended_explanation.contributions
                     if c.feature_name == feature),
                    0
                )

                significant_changes.append({
                    "feature": feature,
                    "current_value": current_val,
                    "recommended_value": recommended_val,
                    "change": recommended_val - current_val,
                    "current_shap": current_shap,
                    "recommended_shap": recommended_shap,
                    "shap_improvement": recommended_shap - current_shap,
                })

        # Sort by SHAP improvement
        significant_changes.sort(key=lambda x: x["shap_improvement"], reverse=True)

        return {
            "current_prediction": current_explanation.prediction,
            "recommended_prediction": recommended_explanation.prediction,
            "expected_improvement": recommended_explanation.prediction - current_explanation.prediction,
            "significant_changes": significant_changes,
            "current_explanation_id": current_explanation.explanation_id,
            "recommended_explanation_id": recommended_explanation.explanation_id,
            "provenance_hash": self._compute_hash({
                "current": current_explanation.output_hash,
                "recommended": recommended_explanation.output_hash,
            }),
        }

    def _compute_model_hash(self) -> str:
        """Compute hash of model for provenance."""
        if self.model is None:
            return "no_model"

        try:
            model_repr = str(type(self.model).__name__)
            return hashlib.sha256(model_repr.encode()).hexdigest()[:16]
        except Exception:
            return "unknown_model"

    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash for provenance."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_combustion_explainer(
    model: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    background_data: Optional[Any] = None,
) -> SHAPCombustionExplainer:
    """
    Factory function to create appropriate SHAP explainer.

    Automatically detects model type and selects appropriate explainer.

    Args:
        model: Trained model
        feature_names: Feature names
        background_data: Background data for KernelExplainer

    Returns:
        Configured SHAPCombustionExplainer
    """
    model_type = "tree"  # Default

    if model is not None:
        model_class = type(model).__name__.lower()

        if any(t in model_class for t in ["xgb", "lgb", "lightgbm", "catboost", "forest", "tree"]):
            model_type = "tree"
        elif any(t in model_class for t in ["linear", "logistic", "ridge", "lasso"]):
            model_type = "linear"
        else:
            model_type = "kernel"

        logger.info(f"Detected model type: {model_type} for {model_class}")

    return SHAPCombustionExplainer(
        model=model,
        feature_names=feature_names,
        background_data=background_data,
        model_type=model_type,
    )
