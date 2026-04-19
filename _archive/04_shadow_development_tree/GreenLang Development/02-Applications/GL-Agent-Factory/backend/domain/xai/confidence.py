# -*- coding: utf-8 -*-
"""
Confidence Scoring Module for GreenLang Agents
==============================================

Provides confidence scoring, calibration, and reliability assessment
for agent outputs to ensure trustworthy predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Categorical confidence levels."""
    VERY_HIGH = "very_high"  # >95% confidence
    HIGH = "high"  # 85-95% confidence
    MEDIUM = "medium"  # 70-85% confidence
    LOW = "low"  # 50-70% confidence
    VERY_LOW = "very_low"  # <50% confidence
    UNCERTAIN = "uncertain"  # Cannot determine


class ReliabilityCategory(str, Enum):
    """Reliability assessment categories."""
    PRODUCTION = "production"  # Ready for production use
    REVIEW = "review"  # Requires human review
    EXPERIMENTAL = "experimental"  # Experimental only
    UNRELIABLE = "unreliable"  # Not reliable


@dataclass
class ConfidenceSource:
    """Single source contributing to confidence."""
    source_name: str
    source_type: str  # "data", "model", "expert", "calculation"
    confidence_contribution: float  # 0-1
    weight: float = 1.0
    notes: Optional[str] = None


class ConfidenceScore(BaseModel):
    """
    Complete confidence assessment for an agent output.

    Combines multiple confidence sources with calibration.
    """
    # Overall confidence
    confidence_value: float = Field(ge=0, le=1)
    confidence_level: ConfidenceLevel
    reliability: ReliabilityCategory

    # Detailed breakdown
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    dominant_source: Optional[str] = None

    # Calibration
    calibrated_confidence: Optional[float] = None
    calibration_applied: bool = False

    # Bounds
    lower_bound: float = 0.0
    upper_bound: float = 1.0

    # Metadata
    sample_size: Optional[int] = None
    calculation_method: str = "weighted_average"

    # Human-readable
    explanation: str = ""
    recommendations: List[str] = Field(default_factory=list)

    @classmethod
    def from_value(cls, value: float, explanation: str = "") -> ConfidenceScore:
        """Create ConfidenceScore from a simple value."""
        level = cls._value_to_level(value)
        reliability = cls._level_to_reliability(level)

        return cls(
            confidence_value=value,
            confidence_level=level,
            reliability=reliability,
            explanation=explanation,
        )

    @staticmethod
    def _value_to_level(value: float) -> ConfidenceLevel:
        """Convert numeric confidence to categorical level."""
        if value >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif value >= 0.85:
            return ConfidenceLevel.HIGH
        elif value >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif value >= 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    @staticmethod
    def _level_to_reliability(level: ConfidenceLevel) -> ReliabilityCategory:
        """Map confidence level to reliability category."""
        mapping = {
            ConfidenceLevel.VERY_HIGH: ReliabilityCategory.PRODUCTION,
            ConfidenceLevel.HIGH: ReliabilityCategory.PRODUCTION,
            ConfidenceLevel.MEDIUM: ReliabilityCategory.REVIEW,
            ConfidenceLevel.LOW: ReliabilityCategory.EXPERIMENTAL,
            ConfidenceLevel.VERY_LOW: ReliabilityCategory.UNRELIABLE,
            ConfidenceLevel.UNCERTAIN: ReliabilityCategory.UNRELIABLE,
        }
        return mapping.get(level, ReliabilityCategory.UNRELIABLE)


@dataclass
class CalibrationPoint:
    """Single point in calibration curve."""
    predicted_confidence: float
    actual_accuracy: float
    sample_count: int


class CalibrationCurve(BaseModel):
    """
    Reliability diagram / calibration curve.

    Maps predicted confidence to actual accuracy for calibration.
    """
    points: List[Dict[str, float]] = Field(default_factory=list)
    n_bins: int = 10

    # Calibration metrics
    expected_calibration_error: float = 0.0  # ECE
    maximum_calibration_error: float = 0.0  # MCE
    average_confidence: float = 0.0
    average_accuracy: float = 0.0

    # Model info
    model_name: str = ""
    calibration_date: Optional[str] = None

    def add_point(self, predicted: float, actual: float, count: int = 1) -> None:
        """Add calibration point."""
        self.points.append({
            "predicted": predicted,
            "actual": actual,
            "count": count,
        })

    def calculate_ece(self) -> float:
        """Calculate Expected Calibration Error."""
        if not self.points:
            return 0.0

        total_samples = sum(p.get("count", 1) for p in self.points)
        if total_samples == 0:
            return 0.0

        ece = 0.0
        for point in self.points:
            weight = point.get("count", 1) / total_samples
            error = abs(point["predicted"] - point["actual"])
            ece += weight * error

        self.expected_calibration_error = ece
        return ece

    def get_calibrated_confidence(self, raw_confidence: float) -> float:
        """Apply calibration to get corrected confidence."""
        if not self.points:
            return raw_confidence

        # Find nearest calibration points
        sorted_points = sorted(self.points, key=lambda p: p["predicted"])

        # Linear interpolation between nearest points
        lower = None
        upper = None

        for point in sorted_points:
            if point["predicted"] <= raw_confidence:
                lower = point
            if point["predicted"] >= raw_confidence and upper is None:
                upper = point

        if lower is None:
            return sorted_points[0]["actual"]
        if upper is None:
            return sorted_points[-1]["actual"]
        if lower == upper:
            return lower["actual"]

        # Interpolate
        t = (raw_confidence - lower["predicted"]) / (upper["predicted"] - lower["predicted"])
        return lower["actual"] + t * (upper["actual"] - lower["actual"])


class ConfidenceCalculator:
    """
    Engine for calculating and calibrating confidence scores.

    Combines multiple confidence sources and applies calibration.
    """

    def __init__(self, calibration: Optional[CalibrationCurve] = None):
        self.calibration = calibration
        self.source_weights: Dict[str, float] = {
            "data_quality": 1.0,
            "model_confidence": 1.0,
            "historical_accuracy": 1.2,
            "expert_override": 1.5,
            "validation_score": 1.0,
        }

    def calculate_confidence(
        self,
        sources: List[ConfidenceSource],
        apply_calibration: bool = True,
    ) -> ConfidenceScore:
        """
        Calculate combined confidence from multiple sources.

        Args:
            sources: List of confidence sources
            apply_calibration: Whether to apply calibration curve

        Returns:
            Complete ConfidenceScore
        """
        if not sources:
            return ConfidenceScore.from_value(0.5, "No confidence sources provided")

        # Weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        source_dicts = []
        max_contribution = 0.0
        dominant = None

        for source in sources:
            effective_weight = source.weight * self.source_weights.get(source.source_type, 1.0)
            contribution = source.confidence_contribution * effective_weight
            weighted_sum += contribution
            total_weight += effective_weight

            source_dict = {
                "name": source.source_name,
                "type": source.source_type,
                "confidence": source.confidence_contribution,
                "weight": effective_weight,
                "contribution": contribution / total_weight if total_weight > 0 else 0,
            }
            source_dicts.append(source_dict)

            if contribution > max_contribution:
                max_contribution = contribution
                dominant = source.source_name

        raw_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5
        raw_confidence = max(0.0, min(1.0, raw_confidence))

        # Apply calibration
        calibrated = None
        if apply_calibration and self.calibration:
            calibrated = self.calibration.get_calibrated_confidence(raw_confidence)

        final_confidence = calibrated if calibrated is not None else raw_confidence
        level = ConfidenceScore._value_to_level(final_confidence)
        reliability = ConfidenceScore._level_to_reliability(level)

        # Generate explanation
        explanation = self._generate_explanation(sources, final_confidence, level)
        recommendations = self._generate_recommendations(level, sources)

        return ConfidenceScore(
            confidence_value=round(final_confidence, 4),
            confidence_level=level,
            reliability=reliability,
            sources=source_dicts,
            dominant_source=dominant,
            calibrated_confidence=calibrated,
            calibration_applied=calibrated is not None,
            explanation=explanation,
            recommendations=recommendations,
            calculation_method="weighted_average",
        )

    def _generate_explanation(
        self,
        sources: List[ConfidenceSource],
        confidence: float,
        level: ConfidenceLevel,
    ) -> str:
        """Generate human-readable explanation."""
        parts = [f"Confidence: {confidence:.1%} ({level.value.replace('_', ' ')})"]

        if sources:
            parts.append("Based on:")
            for source in sorted(sources, key=lambda s: s.confidence_contribution, reverse=True)[:3]:
                parts.append(f"  - {source.source_name}: {source.confidence_contribution:.1%}")

        return "\n".join(parts)

    def _generate_recommendations(
        self,
        level: ConfidenceLevel,
        sources: List[ConfidenceSource],
    ) -> List[str]:
        """Generate recommendations based on confidence level."""
        recommendations = []

        if level in (ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW):
            recommendations.append("Consider gathering additional data to improve confidence")
            recommendations.append("Manual review recommended before using this result")

        if level == ConfidenceLevel.MEDIUM:
            recommendations.append("Result suitable for planning; verify before critical decisions")

        # Check for low individual sources
        for source in sources:
            if source.confidence_contribution < 0.5:
                recommendations.append(f"Improve {source.source_name} ({source.confidence_contribution:.0%})")

        return recommendations[:3]  # Limit to top 3

    def from_accuracy_metrics(
        self,
        accuracy: float,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None,
        sample_size: int = 100,
    ) -> ConfidenceScore:
        """
        Calculate confidence from model accuracy metrics.

        Args:
            accuracy: Overall accuracy (0-1)
            precision: Precision score
            recall: Recall score
            f1_score: F1 score
            sample_size: Number of samples used

        Returns:
            ConfidenceScore based on metrics
        """
        sources = [
            ConfidenceSource(
                source_name="accuracy",
                source_type="model_confidence",
                confidence_contribution=accuracy,
            )
        ]

        if precision is not None:
            sources.append(ConfidenceSource(
                source_name="precision",
                source_type="model_confidence",
                confidence_contribution=precision,
            ))

        if recall is not None:
            sources.append(ConfidenceSource(
                source_name="recall",
                source_type="model_confidence",
                confidence_contribution=recall,
            ))

        if f1_score is not None:
            sources.append(ConfidenceSource(
                source_name="f1_score",
                source_type="model_confidence",
                confidence_contribution=f1_score,
                weight=1.2,  # Slightly higher weight for F1
            ))

        # Adjust for sample size
        size_factor = min(1.0, math.log10(sample_size + 1) / 3)  # Log scale adjustment
        for source in sources:
            source.confidence_contribution *= size_factor

        result = self.calculate_confidence(sources)
        result.sample_size = sample_size
        return result

    def from_data_quality(
        self,
        completeness: float,
        validity: float,
        consistency: float,
        timeliness: float = 1.0,
    ) -> ConfidenceScore:
        """
        Calculate confidence from data quality dimensions.

        Args:
            completeness: Data completeness (0-1)
            validity: Data validity (0-1)
            consistency: Data consistency (0-1)
            timeliness: Data timeliness (0-1)

        Returns:
            ConfidenceScore based on data quality
        """
        sources = [
            ConfidenceSource(
                source_name="completeness",
                source_type="data_quality",
                confidence_contribution=completeness,
            ),
            ConfidenceSource(
                source_name="validity",
                source_type="data_quality",
                confidence_contribution=validity,
                weight=1.2,  # Validity is critical
            ),
            ConfidenceSource(
                source_name="consistency",
                source_type="data_quality",
                confidence_contribution=consistency,
            ),
            ConfidenceSource(
                source_name="timeliness",
                source_type="data_quality",
                confidence_contribution=timeliness,
                weight=0.8,  # Less critical
            ),
        ]

        return self.calculate_confidence(sources)

    def from_emission_factor_tier(
        self,
        tier: int,
        activity_data_confidence: float = 0.9,
        emission_factor_confidence: float = 0.8,
    ) -> ConfidenceScore:
        """
        Calculate confidence for emission calculations by IPCC tier.

        Args:
            tier: IPCC tier (1, 2, or 3)
            activity_data_confidence: Confidence in activity data
            emission_factor_confidence: Confidence in emission factor

        Returns:
            ConfidenceScore based on tier and inputs
        """
        # Base tier confidence
        tier_confidence = {
            1: 0.70,  # Default factors
            2: 0.85,  # Country-specific factors
            3: 0.95,  # Facility-specific data
        }

        base_confidence = tier_confidence.get(tier, 0.70)

        sources = [
            ConfidenceSource(
                source_name=f"IPCC_Tier_{tier}",
                source_type="model_confidence",
                confidence_contribution=base_confidence,
            ),
            ConfidenceSource(
                source_name="activity_data",
                source_type="data_quality",
                confidence_contribution=activity_data_confidence,
            ),
            ConfidenceSource(
                source_name="emission_factor",
                source_type="data_quality",
                confidence_contribution=emission_factor_confidence,
            ),
        ]

        return self.calculate_confidence(sources)


def calculate_confidence(
    value: float = None,
    sources: List[ConfidenceSource] = None,
    accuracy: float = None,
    data_quality: Dict[str, float] = None,
) -> ConfidenceScore:
    """
    High-level function to calculate confidence score.

    Args:
        value: Direct confidence value (0-1)
        sources: List of confidence sources
        accuracy: Model accuracy for automatic calculation
        data_quality: Data quality metrics dict

    Returns:
        ConfidenceScore result
    """
    calculator = ConfidenceCalculator()

    if value is not None:
        return ConfidenceScore.from_value(value)

    if sources:
        return calculator.calculate_confidence(sources)

    if accuracy is not None:
        return calculator.from_accuracy_metrics(accuracy)

    if data_quality:
        return calculator.from_data_quality(
            completeness=data_quality.get("completeness", 1.0),
            validity=data_quality.get("validity", 1.0),
            consistency=data_quality.get("consistency", 1.0),
            timeliness=data_quality.get("timeliness", 1.0),
        )

    return ConfidenceScore.from_value(0.5, "No inputs provided for confidence calculation")


class ConfidenceAggregator:
    """
    Aggregates confidence scores across multiple predictions/outputs.

    Useful for ensemble methods and multi-step calculations.
    """

    def __init__(self):
        self.scores: List[ConfidenceScore] = []

    def add(self, score: ConfidenceScore, weight: float = 1.0) -> None:
        """Add a confidence score to aggregate."""
        # Store with weight as custom attribute
        score_dict = score.model_dump()
        score_dict["_weight"] = weight
        self.scores.append(score)

    def aggregate_weighted_average(self) -> ConfidenceScore:
        """Aggregate using weighted average."""
        if not self.scores:
            return ConfidenceScore.from_value(0.5)

        total_weight = len(self.scores)
        weighted_sum = sum(s.confidence_value for s in self.scores)
        avg_confidence = weighted_sum / total_weight

        return ConfidenceScore.from_value(
            avg_confidence,
            f"Aggregated from {len(self.scores)} scores (weighted average)"
        )

    def aggregate_minimum(self) -> ConfidenceScore:
        """Aggregate using minimum (most conservative)."""
        if not self.scores:
            return ConfidenceScore.from_value(0.5)

        min_confidence = min(s.confidence_value for s in self.scores)

        return ConfidenceScore.from_value(
            min_confidence,
            f"Aggregated from {len(self.scores)} scores (minimum/conservative)"
        )

    def aggregate_product(self) -> ConfidenceScore:
        """Aggregate using product (joint probability assumption)."""
        if not self.scores:
            return ConfidenceScore.from_value(0.5)

        product = 1.0
        for score in self.scores:
            product *= score.confidence_value

        return ConfidenceScore.from_value(
            product,
            f"Aggregated from {len(self.scores)} scores (joint probability)"
        )
