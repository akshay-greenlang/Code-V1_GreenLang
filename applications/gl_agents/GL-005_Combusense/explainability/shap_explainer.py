# -*- coding: utf-8 -*-
"""
SHAP Explainer for GL-005 COMBUSENSE PID Controller Decisions

This module provides explainability for PID controller decisions using
SHAP (SHapley Additive exPlanations) values. It enables operators to understand
why the control system made specific decisions.

Key Features:
    1. TreeExplainer for gradient boosting surrogate models
    2. Feature contribution analysis for control outputs
    3. Real-time explanation generation
    4. Operator-friendly narrative generation
    5. SHA-256 provenance tracking

Use Cases:
    - Explain why control output changed
    - Identify dominant factors in control decisions
    - Generate operator-readable explanations
    - Support regulatory audit requirements
    - Debug unexpected control behavior

Reference:
    - Lundberg, S. M., & Lee, S. (2017). A Unified Approach to Interpreting
      Model Predictions. NIPS 2017.

Example:
    >>> explainer = SHAPExplainer(config)
    >>> explanation = explainer.explain_control_decision(
    ...     features={"setpoint": 1200, "pv": 1180, "error": 20},
    ...     control_output=45.5,
    ...     model=pid_surrogate_model
    ... )
    >>> print(explanation.narrative)
    "Control output of 45.5% driven primarily by error (20C) contributing +12.3%"

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from typing import (
    Dict, List, Optional, Any, Tuple, Union, Callable
)
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import numpy as np
from collections import OrderedDict

# SHAP import with fallback
try:
    import shap
    from shap import TreeExplainer, KernelExplainer, Explainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
    TreeExplainer = None
    KernelExplainer = None

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class ExplainerType(str, Enum):
    """Type of SHAP explainer to use."""
    TREE = "tree"           # For tree-based models (XGBoost, LightGBM)
    KERNEL = "kernel"       # Model-agnostic (slower)
    LINEAR = "linear"       # For linear models
    DEEP = "deep"           # For neural networks
    PERMUTATION = "permutation"  # Permutation-based


class ContributionDirection(str, Enum):
    """Direction of feature contribution."""
    POSITIVE = "positive"   # Feature increases output
    NEGATIVE = "negative"   # Feature decreases output
    NEUTRAL = "neutral"     # Negligible contribution


class ExplanationConfidence(str, Enum):
    """Confidence level of explanation."""
    HIGH = "high"           # Clear, strong contributions
    MEDIUM = "medium"       # Moderate contributions
    LOW = "low"             # Weak or noisy contributions


# =============================================================================
# Pydantic Models
# =============================================================================

class PIDExplainabilityConfig(BaseModel):
    """Configuration for SHAP explainer."""

    # Explainer settings
    explainer_type: ExplainerType = Field(
        default=ExplainerType.TREE,
        description="Type of SHAP explainer"
    )
    n_background_samples: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Background samples for KernelExplainer"
    )

    # Feature settings
    feature_names: List[str] = Field(
        default_factory=lambda: [
            "setpoint", "process_variable", "error",
            "error_integral", "error_derivative",
            "output_previous", "delta_time"
        ],
        description="Names of input features"
    )

    # Thresholds
    significance_threshold: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        description="Minimum contribution to be considered significant"
    )
    top_n_features: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top features to include in explanation"
    )

    # Narrative generation
    generate_narrative: bool = Field(
        default=True,
        description="Generate operator-friendly narrative"
    )
    narrative_language: str = Field(
        default="en",
        description="Language for narrative"
    )

    # Performance
    enable_caching: bool = Field(
        default=True,
        description="Cache explanations for repeated inputs"
    )
    cache_max_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum cache size"
    )


class FeatureContribution(BaseModel):
    """Contribution of a single feature to the prediction."""

    feature_name: str = Field(...)
    feature_value: float = Field(...)
    shap_value: float = Field(...)
    contribution_percent: float = Field(
        ...,
        description="Contribution as percentage of total"
    )
    direction: ContributionDirection = Field(...)
    rank: int = Field(..., ge=1, description="Rank by absolute contribution")
    unit: str = Field(default="", description="Feature unit")

    class Config:
        frozen = True


class SHAPExplanation(BaseModel):
    """Complete SHAP explanation for a control decision."""

    # Identification
    explanation_id: str = Field(...)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Prediction info
    predicted_value: float = Field(...)
    base_value: float = Field(
        ...,
        description="Expected model output (mean of training data)"
    )

    # Feature contributions
    contributions: List[FeatureContribution] = Field(...)
    top_positive_features: List[str] = Field(default_factory=list)
    top_negative_features: List[str] = Field(default_factory=list)

    # Summary statistics
    total_positive_contribution: float = Field(default=0.0)
    total_negative_contribution: float = Field(default=0.0)
    net_contribution: float = Field(default=0.0)

    # Explanation quality
    confidence: ExplanationConfidence = Field(default=ExplanationConfidence.MEDIUM)
    explanation_consistency: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Sum of SHAP values matches prediction difference"
    )

    # Provenance
    model_type: str = Field(default="unknown")
    explainer_type: ExplainerType = Field(default=ExplainerType.TREE)
    provenance_hash: str = Field(default="")


class ControlDecisionExplanation(BaseModel):
    """Complete explanation for a PID control decision."""

    # Identification
    decision_id: str = Field(...)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Control context
    setpoint: float = Field(...)
    process_variable: float = Field(...)
    error: float = Field(...)
    control_output: float = Field(...)
    output_change: float = Field(default=0.0)

    # SHAP explanation
    shap_explanation: SHAPExplanation = Field(...)

    # Narrative
    narrative: str = Field(default="")
    key_insights: List[str] = Field(default_factory=list)

    # Counterfactual
    counterfactual_scenarios: List[Dict[str, Any]] = Field(default_factory=list)

    # Recommendations
    operator_recommendations: List[str] = Field(default_factory=list)

    # Provenance
    provenance_hash: str = Field(default="")


# =============================================================================
# SHAP Explainer Implementation
# =============================================================================

class SHAPExplainer:
    """
    SHAP Explainer for PID Controller Decisions.

    Provides interpretable explanations for control system decisions using
    SHAP (SHapley Additive exPlanations) values. Supports both tree-based
    models (via TreeExplainer) and model-agnostic approaches (via KernelExplainer).

    The explainer works with a surrogate model that approximates the PID
    controller's behavior, allowing SHAP values to be computed even for
    non-ML-based controllers.

    Key Capabilities:
    1. Compute SHAP values for control outputs
    2. Rank features by contribution
    3. Generate operator-friendly narratives
    4. Track provenance for audit

    Example:
        >>> # Initialize explainer
        >>> config = PIDExplainabilityConfig(explainer_type=ExplainerType.TREE)
        >>> explainer = SHAPExplainer(config)
        >>>
        >>> # Train surrogate model on historical PID data
        >>> explainer.fit_surrogate(historical_features, historical_outputs)
        >>>
        >>> # Explain a control decision
        >>> explanation = explainer.explain_control_decision(
        ...     features={"setpoint": 1200, "pv": 1180, "error": 20},
        ...     control_output=45.5
        ... )
        >>>
        >>> # Get narrative
        >>> print(explanation.narrative)
    """

    def __init__(
        self,
        config: Optional[PIDExplainabilityConfig] = None
    ):
        """
        Initialize SHAP Explainer.

        Args:
            config: Explainer configuration
        """
        self.config = config or PIDExplainabilityConfig()

        # SHAP explainer (set after fitting)
        self._explainer: Optional[Any] = None
        self._surrogate_model: Optional[Any] = None
        self._base_value: Optional[float] = None

        # Background data for KernelExplainer
        self._background_data: Optional[np.ndarray] = None

        # Explanation cache
        self._cache: OrderedDict = OrderedDict()

        # Feature units for narrative
        self._feature_units: Dict[str, str] = {
            "setpoint": "C",
            "process_variable": "C",
            "error": "C",
            "error_integral": "C*s",
            "error_derivative": "C/s",
            "output_previous": "%",
            "delta_time": "s",
            "o2_percent": "%",
            "co_ppm": "ppm",
            "flame_intensity": "%",
        }

        # Narrative templates
        self._narrative_templates = {
            "main": (
                "Control output of {output:.1f}% is driven primarily by "
                "{top_feature} ({top_value:.1f}{top_unit}) "
                "contributing {top_contribution:+.1f}%."
            ),
            "secondary": (
                " Secondary factors include {secondary_feature} "
                "({secondary_contribution:+.1f}%)."
            ),
            "stable": (
                "Control output of {output:.1f}% is stable with no "
                "dominant contributing factor."
            ),
            "error_driven": (
                "Control action of {output:.1f}% is primarily responding to "
                "an error of {error:.1f}{error_unit} between setpoint and process variable."
            ),
        }

        if not SHAP_AVAILABLE:
            logger.warning(
                "SHAP library not available. Install with: pip install shap"
            )

        logger.info(
            f"SHAPExplainer initialized: type={self.config.explainer_type.value}, "
            f"features={len(self.config.feature_names)}"
        )

    def fit_surrogate(
        self,
        features: np.ndarray,
        outputs: np.ndarray,
        model_type: str = "xgboost"
    ) -> None:
        """
        Fit a surrogate model to PID controller behavior.

        The surrogate model approximates the relationship between
        input features and control output, enabling SHAP analysis.

        Args:
            features: Historical input features (N x M array)
            outputs: Historical control outputs (N array)
            model_type: Type of surrogate model ("xgboost", "lightgbm", "linear")

        Raises:
            ImportError: If required ML library not available
        """
        logger.info(
            f"Fitting surrogate model: type={model_type}, "
            f"samples={len(outputs)}"
        )

        if model_type == "xgboost":
            try:
                import xgboost as xgb
                self._surrogate_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                self._surrogate_model.fit(features, outputs)
            except ImportError:
                logger.warning("XGBoost not available, using linear model")
                model_type = "linear"

        if model_type == "lightgbm":
            try:
                import lightgbm as lgb
                self._surrogate_model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                self._surrogate_model.fit(features, outputs)
            except ImportError:
                logger.warning("LightGBM not available, using linear model")
                model_type = "linear"

        if model_type == "linear":
            from sklearn.linear_model import Ridge
            self._surrogate_model = Ridge(alpha=1.0)
            self._surrogate_model.fit(features, outputs)

        # Store background data for KernelExplainer
        n_background = min(self.config.n_background_samples, len(features))
        indices = np.random.choice(len(features), n_background, replace=False)
        self._background_data = features[indices]

        # Initialize SHAP explainer
        self._initialize_shap_explainer()

        logger.info(f"Surrogate model fitted: {model_type}")

    def _initialize_shap_explainer(self) -> None:
        """Initialize the appropriate SHAP explainer."""
        if not SHAP_AVAILABLE or self._surrogate_model is None:
            return

        try:
            if self.config.explainer_type == ExplainerType.TREE:
                self._explainer = shap.TreeExplainer(self._surrogate_model)
                self._base_value = self._explainer.expected_value
                if isinstance(self._base_value, np.ndarray):
                    self._base_value = float(self._base_value[0])

            elif self.config.explainer_type == ExplainerType.KERNEL:
                self._explainer = shap.KernelExplainer(
                    self._surrogate_model.predict,
                    self._background_data
                )
                self._base_value = float(np.mean(
                    self._surrogate_model.predict(self._background_data)
                ))

            elif self.config.explainer_type == ExplainerType.LINEAR:
                self._explainer = shap.LinearExplainer(
                    self._surrogate_model,
                    self._background_data
                )
                self._base_value = float(np.mean(
                    self._surrogate_model.predict(self._background_data)
                ))

            logger.info(
                f"SHAP explainer initialized: type={self.config.explainer_type.value}, "
                f"base_value={self._base_value:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self._explainer = None

    def explain_control_decision(
        self,
        features: Union[Dict[str, float], np.ndarray],
        control_output: float,
        setpoint: Optional[float] = None,
        process_variable: Optional[float] = None
    ) -> ControlDecisionExplanation:
        """
        Generate explanation for a PID control decision.

        Args:
            features: Input features (dict or array)
            control_output: The control output to explain
            setpoint: Current setpoint (for narrative)
            process_variable: Current process variable (for narrative)

        Returns:
            Complete explanation with SHAP values and narrative
        """
        # Generate decision ID
        decision_id = self._generate_id(features, control_output)

        # Check cache
        if self.config.enable_caching and decision_id in self._cache:
            logger.debug(f"Returning cached explanation: {decision_id}")
            return self._cache[decision_id]

        # Convert features to array
        if isinstance(features, dict):
            feature_array = self._dict_to_array(features)
            feature_dict = features
        else:
            feature_array = features
            feature_dict = self._array_to_dict(features)

        # Extract setpoint and process variable if not provided
        if setpoint is None:
            setpoint = feature_dict.get("setpoint", 0)
        if process_variable is None:
            process_variable = feature_dict.get("process_variable", 0)
        error = feature_dict.get("error", setpoint - process_variable)

        # Compute SHAP values
        shap_explanation = self._compute_shap_values(
            feature_array,
            feature_dict,
            control_output
        )

        # Generate narrative
        narrative = ""
        key_insights: List[str] = []

        if self.config.generate_narrative:
            narrative, key_insights = self._generate_narrative(
                shap_explanation,
                control_output,
                error
            )

        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(
            feature_dict,
            shap_explanation
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            shap_explanation,
            error,
            control_output
        )

        # Create provenance hash
        provenance_data = {
            "decision_id": decision_id,
            "features": feature_dict,
            "control_output": control_output,
            "shap_values": [c.shap_value for c in shap_explanation.contributions]
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        explanation = ControlDecisionExplanation(
            decision_id=decision_id,
            setpoint=setpoint,
            process_variable=process_variable,
            error=error,
            control_output=control_output,
            shap_explanation=shap_explanation,
            narrative=narrative,
            key_insights=key_insights,
            counterfactual_scenarios=counterfactuals,
            operator_recommendations=recommendations,
            provenance_hash=provenance_hash
        )

        # Cache if enabled
        if self.config.enable_caching:
            self._cache[decision_id] = explanation
            # Limit cache size
            while len(self._cache) > self.config.cache_max_size:
                self._cache.popitem(last=False)

        return explanation

    def _compute_shap_values(
        self,
        feature_array: np.ndarray,
        feature_dict: Dict[str, float],
        control_output: float
    ) -> SHAPExplanation:
        """Compute SHAP values for the given features."""
        explanation_id = hashlib.sha256(
            json.dumps(feature_dict, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Use actual SHAP if available
        if self._explainer is not None and SHAP_AVAILABLE:
            return self._compute_shap_with_library(
                feature_array, feature_dict, control_output, explanation_id
            )

        # Fallback to analytical approximation for PID
        return self._compute_analytical_shap(
            feature_dict, control_output, explanation_id
        )

    def _compute_shap_with_library(
        self,
        feature_array: np.ndarray,
        feature_dict: Dict[str, float],
        control_output: float,
        explanation_id: str
    ) -> SHAPExplanation:
        """Compute SHAP values using the SHAP library."""
        # Reshape for single sample
        if feature_array.ndim == 1:
            feature_array = feature_array.reshape(1, -1)

        # Compute SHAP values
        shap_values = self._explainer.shap_values(feature_array)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = shap_values.flatten()

        # Get base value
        base_value = self._base_value or 0.0

        # Create contributions
        contributions = self._create_contributions(
            feature_dict, shap_values
        )

        # Calculate totals
        total_positive = sum(c.shap_value for c in contributions if c.shap_value > 0)
        total_negative = sum(c.shap_value for c in contributions if c.shap_value < 0)

        # Get top features
        sorted_contributions = sorted(
            contributions,
            key=lambda c: abs(c.shap_value),
            reverse=True
        )
        top_positive = [
            c.feature_name for c in sorted_contributions
            if c.direction == ContributionDirection.POSITIVE
        ][:3]
        top_negative = [
            c.feature_name for c in sorted_contributions
            if c.direction == ContributionDirection.NEGATIVE
        ][:3]

        # Consistency check
        predicted = base_value + sum(shap_values)
        consistency = 1.0 - abs(predicted - control_output) / max(abs(control_output), 1)
        consistency = max(0, min(1, consistency))

        # Determine confidence
        max_contribution = max(abs(c.shap_value) for c in contributions) if contributions else 0
        if max_contribution > self.config.significance_threshold * abs(control_output):
            confidence = ExplanationConfidence.HIGH
        elif max_contribution > self.config.significance_threshold * 0.5 * abs(control_output):
            confidence = ExplanationConfidence.MEDIUM
        else:
            confidence = ExplanationConfidence.LOW

        return SHAPExplanation(
            explanation_id=explanation_id,
            predicted_value=control_output,
            base_value=base_value,
            contributions=contributions,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            total_positive_contribution=total_positive,
            total_negative_contribution=total_negative,
            net_contribution=total_positive + total_negative,
            confidence=confidence,
            explanation_consistency=consistency,
            model_type="surrogate",
            explainer_type=self.config.explainer_type,
            provenance_hash=hashlib.sha256(
                json.dumps({"shap": list(shap_values)}).encode()
            ).hexdigest()
        )

    def _compute_analytical_shap(
        self,
        feature_dict: Dict[str, float],
        control_output: float,
        explanation_id: str
    ) -> SHAPExplanation:
        """
        Compute analytical SHAP-like values for PID controller.

        For a standard PID controller:
            u(t) = Kp*e(t) + Ki*integral(e) + Kd*de/dt

        The "SHAP values" are approximated as the individual term contributions.
        """
        # Extract PID components from features
        error = feature_dict.get("error", 0)
        error_integral = feature_dict.get("error_integral", 0)
        error_derivative = feature_dict.get("error_derivative", 0)

        # Typical PID gains (normalized)
        kp = 1.5
        ki = 0.3
        kd = 0.1

        # Calculate term contributions
        p_contribution = kp * error
        i_contribution = ki * error_integral
        d_contribution = kd * error_derivative

        # Normalize to match control output
        total_contribution = p_contribution + i_contribution + d_contribution
        if abs(total_contribution) > 0.001:
            scale = control_output / total_contribution
        else:
            scale = 1.0

        # Create SHAP-like values
        shap_values = {
            "error": p_contribution * scale,
            "error_integral": i_contribution * scale,
            "error_derivative": d_contribution * scale,
        }

        # Add other features with zero contribution
        for feature in self.config.feature_names:
            if feature not in shap_values:
                shap_values[feature] = 0.0

        # Create contributions list
        contributions: List[FeatureContribution] = []
        total_abs = sum(abs(v) for v in shap_values.values())

        for rank, (feature, shap_value) in enumerate(
            sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True),
            start=1
        ):
            if feature in feature_dict:
                feature_value = feature_dict[feature]
            else:
                feature_value = 0.0

            if abs(shap_value) < 0.001:
                direction = ContributionDirection.NEUTRAL
            elif shap_value > 0:
                direction = ContributionDirection.POSITIVE
            else:
                direction = ContributionDirection.NEGATIVE

            contribution_pct = abs(shap_value) / total_abs * 100 if total_abs > 0 else 0

            contributions.append(FeatureContribution(
                feature_name=feature,
                feature_value=feature_value,
                shap_value=shap_value,
                contribution_percent=contribution_pct,
                direction=direction,
                rank=rank,
                unit=self._feature_units.get(feature, "")
            ))

        # Sort by rank
        contributions.sort(key=lambda c: c.rank)

        # Summary stats
        total_positive = sum(c.shap_value for c in contributions if c.shap_value > 0)
        total_negative = sum(c.shap_value for c in contributions if c.shap_value < 0)

        return SHAPExplanation(
            explanation_id=explanation_id,
            predicted_value=control_output,
            base_value=0.0,
            contributions=contributions,
            top_positive_features=[
                c.feature_name for c in contributions
                if c.direction == ContributionDirection.POSITIVE
            ][:3],
            top_negative_features=[
                c.feature_name for c in contributions
                if c.direction == ContributionDirection.NEGATIVE
            ][:3],
            total_positive_contribution=total_positive,
            total_negative_contribution=total_negative,
            net_contribution=total_positive + total_negative,
            confidence=ExplanationConfidence.MEDIUM,
            explanation_consistency=1.0,
            model_type="pid_analytical",
            explainer_type=ExplainerType.LINEAR,
            provenance_hash=hashlib.sha256(
                json.dumps(shap_values, sort_keys=True).encode()
            ).hexdigest()
        )

    def _create_contributions(
        self,
        feature_dict: Dict[str, float],
        shap_values: np.ndarray
    ) -> List[FeatureContribution]:
        """Create FeatureContribution objects from SHAP values."""
        contributions: List[FeatureContribution] = []
        total_abs = np.sum(np.abs(shap_values))

        feature_names = self.config.feature_names
        if len(feature_names) != len(shap_values):
            feature_names = [f"feature_{i}" for i in range(len(shap_values))]

        for rank, (feature, shap_value) in enumerate(
            sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True),
            start=1
        ):
            shap_value = float(shap_value)

            if abs(shap_value) < 0.001:
                direction = ContributionDirection.NEUTRAL
            elif shap_value > 0:
                direction = ContributionDirection.POSITIVE
            else:
                direction = ContributionDirection.NEGATIVE

            feature_value = feature_dict.get(feature, 0.0)
            contribution_pct = abs(shap_value) / total_abs * 100 if total_abs > 0 else 0

            contributions.append(FeatureContribution(
                feature_name=feature,
                feature_value=feature_value,
                shap_value=shap_value,
                contribution_percent=contribution_pct,
                direction=direction,
                rank=rank,
                unit=self._feature_units.get(feature, "")
            ))

        return contributions

    def _generate_narrative(
        self,
        explanation: SHAPExplanation,
        control_output: float,
        error: float
    ) -> Tuple[str, List[str]]:
        """Generate operator-friendly narrative from SHAP explanation."""
        insights: List[str] = []

        if not explanation.contributions:
            return "No significant contributing factors identified.", insights

        # Get top contributor
        top = explanation.contributions[0]

        # Check if error is the dominant factor (common for PID)
        if top.feature_name in ("error", "error_integral", "error_derivative"):
            narrative = self._narrative_templates["error_driven"].format(
                output=control_output,
                error=error,
                error_unit=self._feature_units.get("error", "")
            )
            insights.append(f"Control action is primarily responding to process error")
        else:
            narrative = self._narrative_templates["main"].format(
                output=control_output,
                top_feature=top.feature_name,
                top_value=top.feature_value,
                top_unit=top.unit,
                top_contribution=top.shap_value
            )

        # Add secondary factor if significant
        if len(explanation.contributions) > 1:
            secondary = explanation.contributions[1]
            if secondary.contribution_percent > 15:
                narrative += self._narrative_templates["secondary"].format(
                    secondary_feature=secondary.feature_name,
                    secondary_contribution=secondary.shap_value
                )
                insights.append(
                    f"Secondary factor: {secondary.feature_name} "
                    f"({secondary.shap_value:+.1f}%)"
                )

        # Add insights for positive contributors
        for c in explanation.contributions[:3]:
            if c.direction == ContributionDirection.POSITIVE and c.contribution_percent > 10:
                insights.append(
                    f"{c.feature_name} is driving output UP by {c.shap_value:.1f}%"
                )
            elif c.direction == ContributionDirection.NEGATIVE and c.contribution_percent > 10:
                insights.append(
                    f"{c.feature_name} is driving output DOWN by {abs(c.shap_value):.1f}%"
                )

        return narrative, insights

    def _generate_counterfactuals(
        self,
        feature_dict: Dict[str, float],
        explanation: SHAPExplanation
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual scenarios."""
        counterfactuals: List[Dict[str, Any]] = []

        for contribution in explanation.contributions[:3]:
            if contribution.contribution_percent < 10:
                continue

            # Create counterfactual by reducing this feature's impact
            cf = {
                "scenario": f"If {contribution.feature_name} were reduced",
                "feature_changed": contribution.feature_name,
                "current_value": contribution.feature_value,
                "hypothetical_value": contribution.feature_value * 0.5,
                "estimated_output_change": -contribution.shap_value * 0.5
            }
            counterfactuals.append(cf)

        return counterfactuals

    def _generate_recommendations(
        self,
        explanation: SHAPExplanation,
        error: float,
        control_output: float
    ) -> List[str]:
        """Generate operator recommendations."""
        recommendations: List[str] = []

        # Check for large error
        if abs(error) > 50:
            recommendations.append(
                f"Large process error ({error:.1f}). Verify setpoint is appropriate."
            )

        # Check for saturation
        if control_output >= 95:
            recommendations.append(
                "Control output near maximum. Check for process constraints."
            )
        elif control_output <= 5:
            recommendations.append(
                "Control output near minimum. Verify process conditions."
            )

        # Check explanation confidence
        if explanation.confidence == ExplanationConfidence.LOW:
            recommendations.append(
                "Explanation confidence is low. Multiple factors may be interacting."
            )

        return recommendations

    def _dict_to_array(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to ordered array."""
        return np.array([
            feature_dict.get(name, 0.0)
            for name in self.config.feature_names
        ])

    def _array_to_dict(self, feature_array: np.ndarray) -> Dict[str, float]:
        """Convert feature array to dictionary."""
        return {
            name: float(feature_array[i])
            for i, name in enumerate(self.config.feature_names)
            if i < len(feature_array)
        }

    def _generate_id(
        self,
        features: Union[Dict[str, float], np.ndarray],
        control_output: float
    ) -> str:
        """Generate unique ID for explanation."""
        if isinstance(features, dict):
            data = {"features": features, "output": control_output}
        else:
            data = {"features": list(features), "output": control_output}

        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def get_feature_importance(self) -> Dict[str, float]:
        """Get global feature importance from surrogate model."""
        if self._surrogate_model is None:
            logger.warning("No surrogate model fitted")
            return {}

        if hasattr(self._surrogate_model, 'feature_importances_'):
            importances = self._surrogate_model.feature_importances_
            return dict(zip(self.config.feature_names, importances))

        return {}

    def clear_cache(self) -> None:
        """Clear explanation cache."""
        self._cache.clear()
        logger.info("SHAP explanation cache cleared")
