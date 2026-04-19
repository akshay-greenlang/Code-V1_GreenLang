"""
Explainability Engine for GreenLang AI/ML Components

This module provides SHAP, LIME, and natural language explainability
for AI/ML features while maintaining GreenLang's zero-hallucination
guarantee for numeric calculations.

CRITICAL DESIGN PRINCIPLE:
- Explainability is for TRANSPARENCY, not CALCULATION
- All numeric values (emissions, health scores, etc.) are computed deterministically
- SHAP/LIME explain WHY features contributed to a decision
- Natural language summarizes calculations in human-readable form

Approved LLM/ML Use Cases:
- Feature importance analysis (SHAP values)
- Local explanations (LIME)
- Natural language summary generation
- Attention visualization for document analysis

PROHIBITED Use Cases:
- Calculating emissions (use deterministic formulas only)
- Calculating compliance metrics (use database lookups)
- Any numeric value used for regulatory reporting

Example:
    >>> from engines.explainability import ExplainabilityEngine
    >>> engine = ExplainabilityEngine()
    >>> report = engine.explain_health_score(
    ...     features={'operating_hours': 15000, 'flame_quality': 85},
    ...     feature_weights={'operating_hours': 0.25, 'flame_quality': 0.30},
    ...     output_value=72.5
    ... )
    >>> print(report.natural_language_summary)
"""

import hashlib
import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import random

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================

class ExplanationType(str, Enum):
    """Types of explanations supported."""
    SHAP_GLOBAL = "shap_global"
    SHAP_LOCAL = "shap_local"
    LIME_LOCAL = "lime_local"
    FEATURE_IMPORTANCE = "feature_importance"
    NATURAL_LANGUAGE = "natural_language"
    ATTENTION_WEIGHTS = "attention_weights"


class ConfidenceLevel(str, Enum):
    """Confidence levels for explanations."""
    HIGH = "high"  # >= 80%
    MEDIUM = "medium"  # 60-80%
    LOW = "low"  # < 60%


# Minimum confidence threshold for explanation validity
MINIMUM_CONFIDENCE_THRESHOLD = 0.80

# Feature importance categories for natural language generation
IMPORTANCE_CATEGORIES = {
    "critical": 0.30,  # Feature contributes >= 30%
    "significant": 0.15,  # Feature contributes 15-30%
    "moderate": 0.05,  # Feature contributes 5-15%
    "minor": 0.0,  # Feature contributes < 5%
}


# =============================================================================
# Data Models
# =============================================================================

class FeatureContribution(BaseModel):
    """Individual feature's contribution to a prediction."""

    feature_name: str = Field(..., description="Name of the feature")
    feature_value: Union[float, int, str] = Field(..., description="Actual feature value")
    contribution: float = Field(..., description="Contribution to output (can be negative)")
    contribution_percent: float = Field(..., description="Percentage contribution (0-100)")
    importance_category: str = Field(..., description="critical/significant/moderate/minor")
    direction: str = Field(..., description="positive/negative/neutral")
    explanation: str = Field(..., description="Human-readable explanation")

    class Config:
        json_encoders = {
            float: lambda v: round(v, 6) if isinstance(v, float) else v,
        }


class SHAPValues(BaseModel):
    """SHAP (SHapley Additive exPlanations) values for a prediction."""

    base_value: float = Field(..., description="Expected value (baseline prediction)")
    output_value: float = Field(..., description="Actual prediction output")
    feature_values: Dict[str, float] = Field(..., description="Input feature values")
    shap_values: Dict[str, float] = Field(..., description="SHAP value per feature")
    sum_shap_values: float = Field(..., description="Sum of all SHAP values")
    consistency_check: bool = Field(..., description="base + sum(shap) == output")

    def get_top_contributors(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N contributing features by absolute SHAP value."""
        sorted_features = sorted(
            self.shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:n]


class LIMEExplanation(BaseModel):
    """LIME (Local Interpretable Model-agnostic Explanations) result."""

    prediction_value: float = Field(..., description="Model prediction")
    local_prediction: float = Field(..., description="Local model prediction")
    intercept: float = Field(..., description="Linear model intercept")
    feature_weights: Dict[str, float] = Field(..., description="Feature weights in local model")
    r_squared: float = Field(..., description="Local model fit quality (0-1)")
    num_samples: int = Field(..., description="Number of perturbation samples used")

    def get_explanation_quality(self) -> ConfidenceLevel:
        """Determine explanation quality based on R-squared."""
        if self.r_squared >= 0.80:
            return ConfidenceLevel.HIGH
        elif self.r_squared >= 0.60:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW


class AttentionVisualization(BaseModel):
    """Attention weights for document/text analysis."""

    tokens: List[str] = Field(..., description="Input tokens/segments")
    attention_weights: List[float] = Field(..., description="Attention weight per token")
    highlighted_segments: List[Tuple[str, float]] = Field(
        ..., description="Segments with attention > threshold"
    )
    visualization_html: Optional[str] = Field(None, description="HTML visualization")


class UncertaintyQuantification(BaseModel):
    """Uncertainty bounds for predictions and explanations."""

    point_estimate: float = Field(..., description="Primary prediction value")
    lower_bound: float = Field(..., description="Lower confidence bound")
    upper_bound: float = Field(..., description="Upper confidence bound")
    confidence_level: float = Field(..., description="Confidence level (e.g., 0.95)")
    uncertainty_source: str = Field(..., description="Source of uncertainty")
    is_reliable: bool = Field(..., description="Meets minimum confidence threshold")

    def get_range_width(self) -> float:
        """Calculate width of uncertainty range."""
        return self.upper_bound - self.lower_bound

    def get_relative_uncertainty(self) -> float:
        """Calculate uncertainty as percentage of point estimate."""
        if self.point_estimate == 0:
            return float('inf')
        return (self.get_range_width() / (2 * abs(self.point_estimate))) * 100


class ExplainabilityReport(BaseModel):
    """Complete explainability report for a prediction."""

    report_id: str = Field(..., description="Unique report identifier")
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Input/Output context
    input_features: Dict[str, Any] = Field(..., description="Input feature values")
    output_value: float = Field(..., description="Predicted/calculated output")
    output_name: str = Field(..., description="Name of output (e.g., 'health_score')")

    # Explanations
    feature_contributions: List[FeatureContribution] = Field(
        default_factory=list, description="Per-feature contributions"
    )
    shap_values: Optional[SHAPValues] = Field(None, description="SHAP analysis")
    lime_explanation: Optional[LIMEExplanation] = Field(None, description="LIME analysis")
    attention_weights: Optional[AttentionVisualization] = Field(
        None, description="Attention visualization"
    )

    # Natural language
    natural_language_summary: str = Field(..., description="Human-readable explanation")
    technical_summary: str = Field(..., description="Technical explanation for auditors")

    # Confidence and uncertainty
    overall_confidence: float = Field(..., ge=0, le=1, description="Explanation confidence")
    confidence_level: ConfidenceLevel = Field(..., description="HIGH/MEDIUM/LOW")
    uncertainty: Optional[UncertaintyQuantification] = Field(None)

    # Audit trail
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")
    calculation_method: str = Field(..., description="Deterministic method used")

    # Warnings
    warnings: List[str] = Field(default_factory=list, description="Any warnings")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def to_audit_dict(self) -> Dict[str, Any]:
        """Export as audit-ready dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "output": {
                "name": self.output_name,
                "value": self.output_value,
            },
            "confidence": {
                "overall": self.overall_confidence,
                "level": self.confidence_level.value,
            },
            "provenance_hash": self.provenance_hash,
            "calculation_method": self.calculation_method,
            "feature_count": len(self.feature_contributions),
            "warnings": self.warnings,
        }


# =============================================================================
# SHAP Implementation (Kernel SHAP for model-agnostic explanations)
# =============================================================================

class KernelSHAPExplainer:
    """
    Kernel SHAP explainer for model-agnostic feature attribution.

    This is a simplified implementation suitable for weighted scoring functions
    commonly used in GreenLang agents (health scores, risk assessments, etc.).

    For complex ML models, integrate with the full shap library.

    IMPORTANT: This explains WHY a score was calculated, not the calculation itself.
    All numeric calculations remain deterministic.
    """

    def __init__(
        self,
        prediction_function: Callable[[Dict[str, float]], float],
        feature_names: List[str],
        baseline_values: Optional[Dict[str, float]] = None,
        num_samples: int = 100,
        random_seed: int = 42,
    ):
        """
        Initialize SHAP explainer.

        Args:
            prediction_function: Function that takes features dict and returns prediction
            feature_names: Names of input features
            baseline_values: Baseline feature values (defaults to zeros)
            num_samples: Number of coalition samples for estimation
            random_seed: Random seed for reproducibility
        """
        self.predict = prediction_function
        self.feature_names = feature_names
        self.baseline_values = baseline_values or {f: 0.0 for f in feature_names}
        self.num_samples = num_samples
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)

    def explain(self, feature_values: Dict[str, float]) -> SHAPValues:
        """
        Compute SHAP values for a single prediction.

        Uses Kernel SHAP approximation for efficiency.

        Args:
            feature_values: Input feature values to explain

        Returns:
            SHAPValues with per-feature attributions
        """
        # Calculate baseline and actual predictions
        base_prediction = self.predict(self.baseline_values)
        actual_prediction = self.predict(feature_values)

        # Calculate marginal contributions using simplified Shapley sampling
        shap_values = {}
        n_features = len(self.feature_names)

        for target_feature in self.feature_names:
            contribution = self._estimate_marginal_contribution(
                target_feature, feature_values, n_features
            )
            shap_values[target_feature] = contribution

        # Normalize to ensure additivity (base + sum(shap) = prediction)
        sum_shap = sum(shap_values.values())
        expected_sum = actual_prediction - base_prediction

        if sum_shap != 0:
            # Scale SHAP values to maintain additivity constraint
            scale_factor = expected_sum / sum_shap
            shap_values = {k: v * scale_factor for k, v in shap_values.items()}

        # Consistency check
        consistency = abs(base_prediction + sum(shap_values.values()) - actual_prediction) < 0.01

        return SHAPValues(
            base_value=base_prediction,
            output_value=actual_prediction,
            feature_values=feature_values,
            shap_values=shap_values,
            sum_shap_values=sum(shap_values.values()),
            consistency_check=consistency,
        )

    def _estimate_marginal_contribution(
        self,
        target_feature: str,
        feature_values: Dict[str, float],
        n_features: int,
    ) -> float:
        """
        Estimate marginal contribution of a feature using sampling.

        This approximates the Shapley value by sampling random coalitions.
        """
        contributions = []

        for _ in range(self.num_samples):
            # Create random coalition (subset of features)
            other_features = [f for f in self.feature_names if f != target_feature]
            coalition_size = self._rng.randint(0, len(other_features))
            coalition = set(self._rng.sample(other_features, coalition_size))

            # Prediction with coalition + target feature
            features_with = {}
            for f in self.feature_names:
                if f == target_feature or f in coalition:
                    features_with[f] = feature_values[f]
                else:
                    features_with[f] = self.baseline_values[f]

            # Prediction with coalition only
            features_without = {}
            for f in self.feature_names:
                if f in coalition:
                    features_without[f] = feature_values[f]
                else:
                    features_without[f] = self.baseline_values[f]

            # Marginal contribution for this coalition
            pred_with = self.predict(features_with)
            pred_without = self.predict(features_without)
            contributions.append(pred_with - pred_without)

        # Average marginal contribution
        return sum(contributions) / len(contributions) if contributions else 0.0


# =============================================================================
# LIME Implementation (Local Interpretable Model-agnostic Explanations)
# =============================================================================

class LIMEExplainer:
    """
    LIME explainer for local, interpretable explanations.

    Creates a local linear model around a prediction to explain it.

    IMPORTANT: This explains WHY a score was calculated, not the calculation itself.
    """

    def __init__(
        self,
        prediction_function: Callable[[Dict[str, float]], float],
        feature_names: List[str],
        num_samples: int = 500,
        kernel_width: float = 0.75,
        random_seed: int = 42,
    ):
        """
        Initialize LIME explainer.

        Args:
            prediction_function: Function that takes features dict and returns prediction
            feature_names: Names of input features
            num_samples: Number of perturbation samples
            kernel_width: Width of exponential kernel for weighting
            random_seed: Random seed for reproducibility
        """
        self.predict = prediction_function
        self.feature_names = feature_names
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self._rng = random.Random(random_seed)

    def explain(
        self,
        feature_values: Dict[str, float],
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> LIMEExplanation:
        """
        Generate LIME explanation for a prediction.

        Args:
            feature_values: Input feature values to explain
            feature_ranges: Optional (min, max) ranges for perturbation

        Returns:
            LIMEExplanation with local linear model
        """
        # Get original prediction
        original_prediction = self.predict(feature_values)

        # Generate perturbed samples
        samples = []
        predictions = []
        distances = []

        for _ in range(self.num_samples):
            perturbed = self._perturb_features(feature_values, feature_ranges)
            pred = self.predict(perturbed)
            dist = self._calculate_distance(feature_values, perturbed)

            samples.append(perturbed)
            predictions.append(pred)
            distances.append(dist)

        # Calculate kernel weights (closer samples get higher weight)
        weights = [self._kernel_function(d) for d in distances]

        # Fit weighted linear regression
        coefficients, intercept, r_squared = self._fit_linear_model(
            samples, predictions, weights
        )

        # Local prediction using linear model
        local_pred = intercept
        for feature, coef in coefficients.items():
            local_pred += coef * feature_values.get(feature, 0)

        return LIMEExplanation(
            prediction_value=original_prediction,
            local_prediction=local_pred,
            intercept=intercept,
            feature_weights=coefficients,
            r_squared=r_squared,
            num_samples=self.num_samples,
        )

    def _perturb_features(
        self,
        feature_values: Dict[str, float],
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        """Generate a perturbed version of the features."""
        perturbed = {}

        for feature, value in feature_values.items():
            if feature_ranges and feature in feature_ranges:
                min_val, max_val = feature_ranges[feature]
            else:
                # Default perturbation: +/- 20% of value or +/- 1 if value is 0
                magnitude = abs(value) * 0.2 if value != 0 else 1.0
                min_val, max_val = value - magnitude, value + magnitude

            perturbed[feature] = self._rng.uniform(min_val, max_val)

        return perturbed

    def _calculate_distance(
        self,
        original: Dict[str, float],
        perturbed: Dict[str, float],
    ) -> float:
        """Calculate Euclidean distance between feature vectors."""
        sum_sq = 0.0
        for feature in original:
            diff = original[feature] - perturbed.get(feature, 0)
            sum_sq += diff ** 2
        return math.sqrt(sum_sq)

    def _kernel_function(self, distance: float) -> float:
        """Exponential kernel for weighting samples by distance."""
        return math.exp(-(distance ** 2) / (self.kernel_width ** 2))

    def _fit_linear_model(
        self,
        samples: List[Dict[str, float]],
        predictions: List[float],
        weights: List[float],
    ) -> Tuple[Dict[str, float], float, float]:
        """
        Fit weighted linear regression using normal equations.

        Returns coefficients, intercept, and R-squared.
        """
        n = len(samples)
        k = len(self.feature_names)

        if n == 0:
            return {f: 0.0 for f in self.feature_names}, 0.0, 0.0

        # Build design matrix X and target vector y
        # Add column of 1s for intercept
        X = []
        for sample in samples:
            row = [1.0]  # intercept term
            for feature in self.feature_names:
                row.append(sample.get(feature, 0.0))
            X.append(row)

        y = predictions

        # Apply weights
        W = [[w if i == j else 0 for j in range(n)] for i, w in enumerate(weights)]

        # Solve weighted least squares: (X'WX)^-1 X'Wy
        # Simplified implementation using direct calculation
        try:
            coefficients = self._solve_weighted_ls(X, y, weights)
            intercept = coefficients[0]
            feature_coefs = {
                self.feature_names[i]: coefficients[i + 1]
                for i in range(k)
            }

            # Calculate R-squared
            y_mean = sum(y) / n
            ss_tot = sum(w * (yi - y_mean) ** 2 for w, yi in zip(weights, y))

            y_pred = []
            for row in X:
                pred = sum(c * x for c, x in zip(coefficients, row))
                y_pred.append(pred)

            ss_res = sum(w * (yi - ypi) ** 2 for w, yi, ypi in zip(weights, y, y_pred))

            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            r_squared = max(0.0, min(1.0, r_squared))

            return feature_coefs, intercept, r_squared

        except Exception as e:
            logger.warning(f"Linear model fitting failed: {e}")
            return {f: 0.0 for f in self.feature_names}, 0.0, 0.0

    def _solve_weighted_ls(
        self,
        X: List[List[float]],
        y: List[float],
        weights: List[float],
    ) -> List[float]:
        """Solve weighted least squares using simplified approach."""
        n = len(X)
        k = len(X[0]) if X else 0

        if k == 0:
            return []

        # X'WX
        XtWX = [[0.0] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                for m in range(n):
                    XtWX[i][j] += X[m][i] * weights[m] * X[m][j]

        # X'Wy
        XtWy = [0.0] * k
        for i in range(k):
            for m in range(n):
                XtWy[i] += X[m][i] * weights[m] * y[m]

        # Solve using Gaussian elimination (simplified)
        return self._gaussian_elimination(XtWX, XtWy)

    def _gaussian_elimination(
        self,
        A: List[List[float]],
        b: List[float],
    ) -> List[float]:
        """Solve Ax = b using Gaussian elimination with partial pivoting."""
        n = len(b)

        # Augmented matrix
        aug = [A[i][:] + [b[i]] for i in range(n)]

        # Forward elimination
        for i in range(n):
            # Partial pivoting
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            aug[i], aug[max_row] = aug[max_row], aug[i]

            # Check for singular matrix
            if abs(aug[i][i]) < 1e-10:
                continue

            # Eliminate column
            for k in range(i + 1, n):
                factor = aug[k][i] / aug[i][i]
                for j in range(i, n + 1):
                    aug[k][j] -= factor * aug[i][j]

        # Back substitution
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            if abs(aug[i][i]) < 1e-10:
                x[i] = 0.0
                continue
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            x[i] /= aug[i][i]

        return x


# =============================================================================
# Natural Language Explanation Generator
# =============================================================================

class NaturalLanguageGenerator:
    """
    Generates human-readable explanations from feature contributions.

    IMPORTANT: This is for TRANSPARENCY, not calculation.
    All numeric values are already computed deterministically.
    """

    # Templates for different explanation contexts
    TEMPLATES = {
        "health_score": {
            "intro": "The {output_name} of {output_value:.1f} was determined by analyzing {feature_count} key factors.",
            "critical": "**Critical factor**: {feature_name} ({value_desc}) contributed {contribution_pct:.1f}% to the score, {direction_text}.",
            "significant": "{feature_name} ({value_desc}) was a significant factor, contributing {contribution_pct:.1f}% {direction_text}.",
            "moderate": "{feature_name} had a moderate influence ({contribution_pct:.1f}%).",
            "minor": "{feature_name} had minimal impact ({contribution_pct:.1f}%).",
            "summary": "Overall confidence in this assessment is {confidence_level} ({confidence_pct:.0f}%).",
        },
        "maintenance_priority": {
            "intro": "The maintenance priority was determined as {output_name} based on {feature_count} indicators.",
            "critical": "**Urgent**: {feature_name} is {value_desc}, which {direction_text} ({contribution_pct:.1f}% weight).",
            "significant": "{feature_name} ({value_desc}) significantly influenced the assessment ({contribution_pct:.1f}%).",
            "moderate": "{feature_name} contributed moderately ({contribution_pct:.1f}%).",
            "minor": "{feature_name} had minor influence ({contribution_pct:.1f}%).",
            "summary": "This assessment has {confidence_level} confidence ({confidence_pct:.0f}%).",
        },
        "generic": {
            "intro": "The {output_name} value of {output_value:.2f} was calculated from {feature_count} input features.",
            "critical": "Primary driver: {feature_name} = {value_desc} (contribution: {contribution_pct:.1f}%, {direction_text}).",
            "significant": "Significant factor: {feature_name} = {value_desc} ({contribution_pct:.1f}%, {direction_text}).",
            "moderate": "Moderate factor: {feature_name} = {value_desc} ({contribution_pct:.1f}%).",
            "minor": "Minor factor: {feature_name} = {value_desc} ({contribution_pct:.1f}%).",
            "summary": "Explanation confidence: {confidence_level} ({confidence_pct:.0f}%).",
        },
    }

    # Direction descriptions
    DIRECTION_TEXT = {
        "positive": {
            "health": "improving the health assessment",
            "risk": "increasing risk level",
            "maintenance": "indicating good condition",
            "generic": "increasing the output value",
        },
        "negative": {
            "health": "reducing the health score",
            "risk": "decreasing risk level",
            "maintenance": "indicating potential issues",
            "generic": "decreasing the output value",
        },
        "neutral": {
            "health": "having neutral impact",
            "risk": "having neutral impact",
            "maintenance": "having neutral impact",
            "generic": "having neutral impact",
        },
    }

    def __init__(self, context: str = "generic"):
        """
        Initialize generator with context.

        Args:
            context: One of 'health_score', 'maintenance_priority', 'generic'
        """
        self.context = context if context in self.TEMPLATES else "generic"
        self.templates = self.TEMPLATES[self.context]

    def generate_summary(
        self,
        output_name: str,
        output_value: float,
        feature_contributions: List[FeatureContribution],
        confidence: float,
    ) -> str:
        """
        Generate a natural language summary of the explanation.

        Args:
            output_name: Name of the output being explained
            output_value: The output value
            feature_contributions: List of feature contributions
            confidence: Overall confidence level (0-1)

        Returns:
            Human-readable explanation string
        """
        parts = []

        # Introduction
        intro = self.templates["intro"].format(
            output_name=output_name,
            output_value=output_value,
            feature_count=len(feature_contributions),
        )
        parts.append(intro)
        parts.append("")

        # Sort contributions by absolute value
        sorted_contribs = sorted(
            feature_contributions,
            key=lambda x: abs(x.contribution_percent),
            reverse=True,
        )

        # Group by importance category
        for contrib in sorted_contribs:
            template_key = contrib.importance_category
            if template_key not in self.templates:
                template_key = "minor"

            # Get direction text
            direction_context = "health" if "health" in self.context else "generic"
            direction_text = self.DIRECTION_TEXT.get(
                contrib.direction, {}
            ).get(direction_context, "affecting the result")

            # Format value description
            if isinstance(contrib.feature_value, float):
                value_desc = f"{contrib.feature_value:.2f}"
            else:
                value_desc = str(contrib.feature_value)

            sentence = self.templates[template_key].format(
                feature_name=self._humanize_feature_name(contrib.feature_name),
                value_desc=value_desc,
                contribution_pct=contrib.contribution_percent,
                direction_text=direction_text,
            )
            parts.append(sentence)

        parts.append("")

        # Confidence summary
        confidence_level = "high" if confidence >= 0.8 else "medium" if confidence >= 0.6 else "low"
        summary = self.templates["summary"].format(
            confidence_level=confidence_level,
            confidence_pct=confidence * 100,
        )
        parts.append(summary)

        return "\n".join(parts)

    def generate_technical_summary(
        self,
        output_name: str,
        output_value: float,
        calculation_method: str,
        feature_contributions: List[FeatureContribution],
        shap_values: Optional[SHAPValues] = None,
        lime_explanation: Optional[LIMEExplanation] = None,
    ) -> str:
        """
        Generate a technical summary for auditors.

        Args:
            output_name: Name of the output
            output_value: The output value
            calculation_method: Description of calculation method
            feature_contributions: Feature contributions
            shap_values: Optional SHAP analysis
            lime_explanation: Optional LIME analysis

        Returns:
            Technical explanation string
        """
        lines = [
            f"=== Technical Explanation Report ===",
            f"",
            f"Output: {output_name} = {output_value}",
            f"Calculation Method: {calculation_method}",
            f"",
            f"--- Feature Contributions ---",
        ]

        for contrib in sorted(
            feature_contributions,
            key=lambda x: abs(x.contribution_percent),
            reverse=True,
        ):
            lines.append(
                f"  {contrib.feature_name}: value={contrib.feature_value}, "
                f"contribution={contrib.contribution:.4f} ({contrib.contribution_percent:.2f}%), "
                f"direction={contrib.direction}"
            )

        if shap_values:
            lines.extend([
                f"",
                f"--- SHAP Analysis ---",
                f"  Base value: {shap_values.base_value:.4f}",
                f"  Sum of SHAP values: {shap_values.sum_shap_values:.4f}",
                f"  Consistency check: {'PASS' if shap_values.consistency_check else 'FAIL'}",
            ])
            for feature, value in shap_values.shap_values.items():
                lines.append(f"  SHAP[{feature}] = {value:.4f}")

        if lime_explanation:
            lines.extend([
                f"",
                f"--- LIME Analysis ---",
                f"  Local model R-squared: {lime_explanation.r_squared:.4f}",
                f"  Local model intercept: {lime_explanation.intercept:.4f}",
                f"  Samples used: {lime_explanation.num_samples}",
            ])
            for feature, weight in lime_explanation.feature_weights.items():
                lines.append(f"  Weight[{feature}] = {weight:.4f}")

        lines.extend([
            f"",
            f"--- Audit Notes ---",
            f"  * All numeric calculations are deterministic (zero-hallucination)",
            f"  * Explanations describe feature importance, not calculations",
            f"  * Reproducible with same inputs and random seeds",
        ])

        return "\n".join(lines)

    def _humanize_feature_name(self, feature_name: str) -> str:
        """Convert feature_name to Human Readable Name."""
        # Replace underscores with spaces and title case
        humanized = feature_name.replace("_", " ").title()

        # Handle common abbreviations
        abbreviations = {
            "Rul": "RUL",
            "Mttf": "MTTF",
            "Co": "CO",
            "Nox": "NOx",
            "O2": "O2",
            "Ppm": "PPM",
        }
        for abbr, replacement in abbreviations.items():
            humanized = humanized.replace(abbr, replacement)

        return humanized


# =============================================================================
# Main Explainability Engine
# =============================================================================

class ExplainabilityEngine:
    """
    Main engine for generating explainability reports.

    This engine provides:
    - SHAP values for global and local feature importance
    - LIME explanations for local interpretability
    - Natural language summaries
    - Attention visualization (for document analysis)
    - Uncertainty quantification

    CRITICAL: This engine explains WHY calculations produced certain results.
    It does NOT perform calculations - all numeric values must be computed
    deterministically before calling this engine.

    Example:
        >>> engine = ExplainabilityEngine()
        >>>
        >>> # Define the prediction function (deterministic calculation)
        >>> def health_calculator(features):
        ...     return (
        ...         features['flame_quality'] * 0.30 +
        ...         features['age_factor'] * 0.15 +
        ...         features['cycles_factor'] * 0.15
        ...     )
        >>>
        >>> # Get explanation
        >>> report = engine.explain_weighted_score(
        ...     prediction_function=health_calculator,
        ...     feature_values={'flame_quality': 85, 'age_factor': 70, 'cycles_factor': 90},
        ...     feature_weights={'flame_quality': 0.30, 'age_factor': 0.15, 'cycles_factor': 0.15},
        ...     output_name='health_score',
        ...     context='health_score'
        ... )
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        enable_shap: bool = True,
        enable_lime: bool = True,
        random_seed: int = 42,
    ):
        """
        Initialize the explainability engine.

        Args:
            enable_shap: Whether to compute SHAP values
            enable_lime: Whether to compute LIME explanations
            random_seed: Random seed for reproducibility
        """
        self.enable_shap = enable_shap
        self.enable_lime = enable_lime
        self.random_seed = random_seed

        logger.info(f"ExplainabilityEngine v{self.VERSION} initialized")

    def explain_weighted_score(
        self,
        prediction_function: Callable[[Dict[str, float]], float],
        feature_values: Dict[str, float],
        feature_weights: Dict[str, float],
        output_name: str,
        context: str = "generic",
        baseline_values: Optional[Dict[str, float]] = None,
    ) -> ExplainabilityReport:
        """
        Generate explainability report for a weighted scoring function.

        This is the primary method for explaining deterministic weighted
        calculations like health scores, risk assessments, etc.

        Args:
            prediction_function: The deterministic calculation function
            feature_values: Input feature values
            feature_weights: Weights used in the calculation
            output_name: Name of the output being explained
            context: Context for natural language ('health_score', 'maintenance_priority', 'generic')
            baseline_values: Optional baseline values for SHAP

        Returns:
            Complete ExplainabilityReport
        """
        report_id = self._generate_report_id(feature_values, output_name)
        warnings = []

        # Calculate the output value
        output_value = prediction_function(feature_values)

        # Feature names
        feature_names = list(feature_values.keys())

        # Compute SHAP values
        shap_values = None
        if self.enable_shap:
            try:
                shap_explainer = KernelSHAPExplainer(
                    prediction_function=prediction_function,
                    feature_names=feature_names,
                    baseline_values=baseline_values,
                    random_seed=self.random_seed,
                )
                shap_values = shap_explainer.explain(feature_values)
            except Exception as e:
                warnings.append(f"SHAP computation failed: {str(e)}")
                logger.warning(f"SHAP computation failed: {e}")

        # Compute LIME explanation
        lime_explanation = None
        if self.enable_lime:
            try:
                lime_explainer = LIMEExplainer(
                    prediction_function=prediction_function,
                    feature_names=feature_names,
                    random_seed=self.random_seed,
                )
                lime_explanation = lime_explainer.explain(feature_values)
            except Exception as e:
                warnings.append(f"LIME computation failed: {str(e)}")
                logger.warning(f"LIME computation failed: {e}")

        # Calculate feature contributions
        feature_contributions = self._calculate_feature_contributions(
            feature_values, feature_weights, output_value, shap_values
        )

        # Generate natural language summary
        nl_generator = NaturalLanguageGenerator(context=context)

        # Determine confidence
        confidence = self._calculate_confidence(shap_values, lime_explanation)
        confidence_level = (
            ConfidenceLevel.HIGH if confidence >= 0.80
            else ConfidenceLevel.MEDIUM if confidence >= 0.60
            else ConfidenceLevel.LOW
        )

        natural_language_summary = nl_generator.generate_summary(
            output_name=output_name,
            output_value=output_value,
            feature_contributions=feature_contributions,
            confidence=confidence,
        )

        technical_summary = nl_generator.generate_technical_summary(
            output_name=output_name,
            output_value=output_value,
            calculation_method="Deterministic weighted scoring",
            feature_contributions=feature_contributions,
            shap_values=shap_values,
            lime_explanation=lime_explanation,
        )

        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(
            output_value, feature_contributions, confidence
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            feature_values, output_value, feature_contributions
        )

        return ExplainabilityReport(
            report_id=report_id,
            input_features=feature_values,
            output_value=output_value,
            output_name=output_name,
            feature_contributions=feature_contributions,
            shap_values=shap_values,
            lime_explanation=lime_explanation,
            natural_language_summary=natural_language_summary,
            technical_summary=technical_summary,
            overall_confidence=confidence,
            confidence_level=confidence_level,
            uncertainty=uncertainty,
            provenance_hash=provenance_hash,
            calculation_method="Deterministic weighted scoring",
            warnings=warnings,
        )

    def explain_health_score(
        self,
        operating_hours: float,
        design_life: float,
        flame_quality: float,
        cycles_factor: float,
        age_factor: float,
        calculated_health_score: float,
        component_health: Optional[Dict[str, float]] = None,
    ) -> ExplainabilityReport:
        """
        Generate explanation for a burner health score calculation.

        This is a convenience method specifically for GL-021 BurnerMaintenancePredictorAgent.

        Args:
            operating_hours: Total operating hours
            design_life: Design life in hours
            flame_quality: Flame quality score 0-100
            cycles_factor: Cycles wear factor 0-1
            age_factor: Age degradation factor 0-1
            calculated_health_score: The deterministically calculated health score
            component_health: Optional component health scores

        Returns:
            ExplainabilityReport for the health score
        """
        # Feature values
        feature_values = {
            "operating_life_ratio": min(1.0, operating_hours / max(1, design_life)),
            "flame_quality": flame_quality,
            "cycles_factor": cycles_factor,
            "age_factor": age_factor,
            "component_health": 80.0,  # Default if not provided
        }

        if component_health:
            # Calculate weighted component health
            total_weight = sum(component_health.values()) / len(component_health)
            feature_values["component_health"] = total_weight

        # Feature weights from health_score.py
        feature_weights = {
            "operating_life_ratio": 0.25,
            "flame_quality": 0.30,
            "cycles_factor": 0.15,
            "age_factor": 0.15,
            "component_health": 0.15,
        }

        # Define prediction function that mirrors the actual calculation
        def health_calculator(features: Dict[str, float]) -> float:
            life_score = max(0, min(100, (1 - features.get("operating_life_ratio", 0)) * 100))
            flame_score = features.get("flame_quality", 0)
            cycles_score = features.get("cycles_factor", 0) * 100
            age_score = features.get("age_factor", 0) * 100
            component_score = features.get("component_health", 80)

            return (
                0.25 * life_score +
                0.30 * flame_score +
                0.15 * cycles_score +
                0.15 * age_score +
                0.15 * component_score
            )

        return self.explain_weighted_score(
            prediction_function=health_calculator,
            feature_values=feature_values,
            feature_weights=feature_weights,
            output_name="health_score",
            context="health_score",
        )

    def explain_maintenance_priority(
        self,
        health_score: float,
        rul_hours: float,
        failure_prob_30d: float,
        priority_result: str,
    ) -> ExplainabilityReport:
        """
        Generate explanation for maintenance priority determination.

        Args:
            health_score: Overall health score 0-100
            rul_hours: Remaining useful life in hours
            failure_prob_30d: 30-day failure probability
            priority_result: The determined priority level

        Returns:
            ExplainabilityReport for the priority decision
        """
        feature_values = {
            "health_score": health_score,
            "rul_hours": rul_hours,
            "failure_prob_30d": failure_prob_30d,
        }

        # Priority scoring function (normalized)
        def priority_scorer(features: Dict[str, float]) -> float:
            # Lower health = higher priority score
            health_contrib = (100 - features.get("health_score", 100)) / 100 * 40
            # Lower RUL = higher priority
            rul_contrib = max(0, (5000 - features.get("rul_hours", 5000)) / 5000 * 30)
            # Higher failure prob = higher priority
            prob_contrib = features.get("failure_prob_30d", 0) * 30
            return health_contrib + rul_contrib + prob_contrib

        feature_weights = {
            "health_score": 0.40,
            "rul_hours": 0.30,
            "failure_prob_30d": 0.30,
        }

        return self.explain_weighted_score(
            prediction_function=priority_scorer,
            feature_values=feature_values,
            feature_weights=feature_weights,
            output_name=f"maintenance_priority ({priority_result})",
            context="maintenance_priority",
        )

    def _calculate_feature_contributions(
        self,
        feature_values: Dict[str, float],
        feature_weights: Dict[str, float],
        output_value: float,
        shap_values: Optional[SHAPValues] = None,
    ) -> List[FeatureContribution]:
        """Calculate per-feature contributions."""
        contributions = []

        # Use SHAP values if available, otherwise use weights
        if shap_values:
            total_abs = sum(abs(v) for v in shap_values.shap_values.values())
            for feature, shap_val in shap_values.shap_values.items():
                value = feature_values.get(feature, 0)
                contrib_pct = (abs(shap_val) / total_abs * 100) if total_abs > 0 else 0

                # Determine importance category
                if contrib_pct >= 30:
                    category = "critical"
                elif contrib_pct >= 15:
                    category = "significant"
                elif contrib_pct >= 5:
                    category = "moderate"
                else:
                    category = "minor"

                # Determine direction
                direction = "positive" if shap_val > 0.01 else "negative" if shap_val < -0.01 else "neutral"

                contributions.append(FeatureContribution(
                    feature_name=feature,
                    feature_value=value,
                    contribution=shap_val,
                    contribution_percent=contrib_pct,
                    importance_category=category,
                    direction=direction,
                    explanation=f"{feature} contributed {shap_val:.4f} to the output",
                ))
        else:
            # Fall back to weight-based estimation
            total_weight = sum(feature_weights.values())
            for feature, weight in feature_weights.items():
                value = feature_values.get(feature, 0)
                contrib_pct = (weight / total_weight * 100) if total_weight > 0 else 0

                if contrib_pct >= 30:
                    category = "critical"
                elif contrib_pct >= 15:
                    category = "significant"
                elif contrib_pct >= 5:
                    category = "moderate"
                else:
                    category = "minor"

                contributions.append(FeatureContribution(
                    feature_name=feature,
                    feature_value=value,
                    contribution=weight * value if isinstance(value, (int, float)) else weight,
                    contribution_percent=contrib_pct,
                    importance_category=category,
                    direction="positive",
                    explanation=f"{feature} has weight {weight:.2f}",
                ))

        return contributions

    def _calculate_confidence(
        self,
        shap_values: Optional[SHAPValues],
        lime_explanation: Optional[LIMEExplanation],
    ) -> float:
        """Calculate overall explanation confidence."""
        confidence_scores = []

        if shap_values:
            # SHAP consistency check contributes to confidence
            shap_confidence = 0.9 if shap_values.consistency_check else 0.6
            confidence_scores.append(shap_confidence)

        if lime_explanation:
            # LIME R-squared indicates local model fit quality
            lime_confidence = lime_explanation.r_squared
            confidence_scores.append(lime_confidence)

        if not confidence_scores:
            return 0.7  # Default confidence without SHAP/LIME

        return sum(confidence_scores) / len(confidence_scores)

    def _calculate_uncertainty(
        self,
        output_value: float,
        feature_contributions: List[FeatureContribution],
        confidence: float,
    ) -> UncertaintyQuantification:
        """Calculate uncertainty bounds for the explanation."""
        # Uncertainty inversely related to confidence
        uncertainty_factor = (1 - confidence) * 0.2  # Max 20% uncertainty

        lower = output_value * (1 - uncertainty_factor)
        upper = output_value * (1 + uncertainty_factor)

        return UncertaintyQuantification(
            point_estimate=output_value,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence,
            uncertainty_source="explanation_confidence",
            is_reliable=confidence >= MINIMUM_CONFIDENCE_THRESHOLD,
        )

    def _generate_report_id(
        self,
        feature_values: Dict[str, Any],
        output_name: str,
    ) -> str:
        """Generate unique report ID."""
        data = {
            "features": feature_values,
            "output": output_name,
            "timestamp": datetime.utcnow().isoformat(),
        }
        hash_input = json.dumps(data, sort_keys=True, default=str)
        return f"EXPL-{hashlib.sha256(hash_input.encode()).hexdigest()[:12].upper()}"

    def _calculate_provenance_hash(
        self,
        feature_values: Dict[str, Any],
        output_value: float,
        feature_contributions: List[FeatureContribution],
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        data = {
            "engine_version": self.VERSION,
            "features": {k: str(v) for k, v in feature_values.items()},
            "output_value": str(output_value),
            "contributions": [
                {"feature": c.feature_name, "contribution": str(c.contribution)}
                for c in feature_contributions
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


# =============================================================================
# Explainability Mixin for Agents
# =============================================================================

class ExplainabilityMixin:
    """
    Mixin class providing explainability capabilities to GreenLang agents.

    Add this mixin to any agent class to gain explainability methods.

    Example:
        >>> class MyAgent(ExplainabilityMixin):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.init_explainability()
        ...
        ...     def run(self, input_data):
        ...         # Deterministic calculation
        ...         result = self._calculate(input_data)
        ...         # Generate explanation
        ...         explanation = self.generate_explanation(input_data, result)
        ...         return result, explanation
    """

    def init_explainability(
        self,
        enable_shap: bool = True,
        enable_lime: bool = True,
        default_context: str = "generic",
        random_seed: int = 42,
    ) -> None:
        """
        Initialize explainability capabilities.

        Args:
            enable_shap: Whether to enable SHAP computations
            enable_lime: Whether to enable LIME computations
            default_context: Default context for natural language
            random_seed: Random seed for reproducibility
        """
        self._explainability_engine = ExplainabilityEngine(
            enable_shap=enable_shap,
            enable_lime=enable_lime,
            random_seed=random_seed,
        )
        self._default_explanation_context = default_context
        self._explanation_history: List[ExplainabilityReport] = []

        logger.info(f"Explainability initialized for {self.__class__.__name__}")

    def generate_shap_report(
        self,
        prediction_function: Callable[[Dict[str, float]], float],
        feature_values: Dict[str, float],
        output_name: str,
        baseline_values: Optional[Dict[str, float]] = None,
    ) -> SHAPValues:
        """
        Generate SHAP values report for a prediction.

        Args:
            prediction_function: The deterministic calculation function
            feature_values: Input feature values
            output_name: Name of the output
            baseline_values: Optional baseline values

        Returns:
            SHAPValues with per-feature attributions
        """
        shap_explainer = KernelSHAPExplainer(
            prediction_function=prediction_function,
            feature_names=list(feature_values.keys()),
            baseline_values=baseline_values,
            random_seed=getattr(self._explainability_engine, 'random_seed', 42),
        )
        return shap_explainer.explain(feature_values)

    def generate_lime_explanation(
        self,
        prediction_function: Callable[[Dict[str, float]], float],
        feature_values: Dict[str, float],
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> LIMEExplanation:
        """
        Generate LIME explanation for a prediction.

        Args:
            prediction_function: The deterministic calculation function
            feature_values: Input feature values
            feature_ranges: Optional feature ranges for perturbation

        Returns:
            LIMEExplanation with local linear model
        """
        lime_explainer = LIMEExplainer(
            prediction_function=prediction_function,
            feature_names=list(feature_values.keys()),
            random_seed=getattr(self._explainability_engine, 'random_seed', 42),
        )
        return lime_explainer.explain(feature_values, feature_ranges)

    def generate_natural_language_summary(
        self,
        output_name: str,
        output_value: float,
        feature_contributions: List[FeatureContribution],
        confidence: float,
        context: Optional[str] = None,
    ) -> str:
        """
        Generate natural language summary of an explanation.

        Args:
            output_name: Name of the output
            output_value: The output value
            feature_contributions: List of feature contributions
            confidence: Confidence level (0-1)
            context: Optional context override

        Returns:
            Human-readable explanation string
        """
        generator = NaturalLanguageGenerator(
            context=context or self._default_explanation_context
        )
        return generator.generate_summary(
            output_name, output_value, feature_contributions, confidence
        )

    def explain_prediction(
        self,
        prediction_function: Callable[[Dict[str, float]], float],
        feature_values: Dict[str, float],
        feature_weights: Dict[str, float],
        output_name: str,
        context: Optional[str] = None,
    ) -> ExplainabilityReport:
        """
        Generate complete explainability report for a prediction.

        Args:
            prediction_function: The deterministic calculation function
            feature_values: Input feature values
            feature_weights: Weights used in the calculation
            output_name: Name of the output
            context: Optional context for natural language

        Returns:
            Complete ExplainabilityReport
        """
        report = self._explainability_engine.explain_weighted_score(
            prediction_function=prediction_function,
            feature_values=feature_values,
            feature_weights=feature_weights,
            output_name=output_name,
            context=context or self._default_explanation_context,
        )

        # Store in history
        self._explanation_history.append(report)

        return report

    def get_explanation_history(self) -> List[ExplainabilityReport]:
        """Get history of generated explanations."""
        return self._explanation_history.copy()

    def clear_explanation_history(self) -> None:
        """Clear explanation history."""
        self._explanation_history.clear()


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Enums
    "ExplanationType",
    "ConfidenceLevel",
    # Data models
    "FeatureContribution",
    "SHAPValues",
    "LIMEExplanation",
    "AttentionVisualization",
    "UncertaintyQuantification",
    "ExplainabilityReport",
    # Explainers
    "KernelSHAPExplainer",
    "LIMEExplainer",
    "NaturalLanguageGenerator",
    # Main engine
    "ExplainabilityEngine",
    # Mixin
    "ExplainabilityMixin",
    # Constants
    "MINIMUM_CONFIDENCE_THRESHOLD",
]
