# -*- coding: utf-8 -*-
"""
Confidence Reporter Module

This module provides a unified API for reporting prediction confidence
and uncertainty in GreenLang ML models, aggregating information from
ensemble predictions, conformal intervals, and calibration.

The confidence reporter provides a single interface for stakeholders
to understand prediction reliability, critical for regulatory compliance
and decision-making.

Example:
    >>> from greenlang.ml.uncertainty import ConfidenceReporter
    >>> reporter = ConfidenceReporter(ensemble, conformal, calibrator)
    >>> report = reporter.generate_report(X_test, y_pred)
    >>> print(f"Overall confidence: {report.overall_confidence:.2%}")
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence level classifications."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class UncertaintySource(str, Enum):
    """Sources of uncertainty."""
    ALEATORIC = "aleatoric"  # Data uncertainty
    EPISTEMIC = "epistemic"  # Model uncertainty
    COMBINED = "combined"


class ConfidenceReporterConfig(BaseModel):
    """Configuration for confidence reporter."""

    confidence_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "very_high": 0.95,
            "high": 0.85,
            "medium": 0.70,
            "low": 0.50
        },
        description="Thresholds for confidence levels"
    )
    include_intervals: bool = Field(
        default=True,
        description="Include prediction intervals"
    )
    include_calibration: bool = Field(
        default=True,
        description="Include calibration information"
    )
    include_ensemble_agreement: bool = Field(
        default=True,
        description="Include ensemble agreement"
    )
    interval_confidence: float = Field(
        default=0.95,
        description="Confidence level for intervals"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )


class PredictionConfidence(BaseModel):
    """Confidence information for a single prediction."""

    prediction: float = Field(
        ...,
        description="Point prediction"
    )
    confidence_score: float = Field(
        ...,
        description="Overall confidence score (0-1)"
    )
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Categorical confidence level"
    )
    lower_bound: Optional[float] = Field(
        default=None,
        description="Lower prediction interval"
    )
    upper_bound: Optional[float] = Field(
        default=None,
        description="Upper prediction interval"
    )
    interval_width: Optional[float] = Field(
        default=None,
        description="Prediction interval width"
    )
    ensemble_std: Optional[float] = Field(
        default=None,
        description="Ensemble standard deviation"
    )
    ensemble_agreement: Optional[float] = Field(
        default=None,
        description="Ensemble agreement score"
    )
    calibrated_probability: Optional[float] = Field(
        default=None,
        description="Calibrated probability"
    )
    uncertainty_sources: Dict[str, float] = Field(
        default_factory=dict,
        description="Decomposed uncertainty sources"
    )
    recommendation: str = Field(
        default="",
        description="Action recommendation"
    )


class ConfidenceReport(BaseModel):
    """Comprehensive confidence report."""

    predictions: List[PredictionConfidence] = Field(
        ...,
        description="Per-prediction confidence"
    )
    overall_confidence: float = Field(
        ...,
        description="Average confidence score"
    )
    confidence_distribution: Dict[str, int] = Field(
        ...,
        description="Distribution of confidence levels"
    )
    high_confidence_ratio: float = Field(
        ...,
        description="Ratio of high confidence predictions"
    )
    avg_interval_width: Optional[float] = Field(
        default=None,
        description="Average interval width"
    )
    avg_ensemble_agreement: Optional[float] = Field(
        default=None,
        description="Average ensemble agreement"
    )
    calibration_ece: Optional[float] = Field(
        default=None,
        description="Expected Calibration Error"
    )
    summary: str = Field(
        ...,
        description="Human-readable summary"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Overall recommendations"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Report timestamp"
    )


class ConfidenceReporter:
    """
    Confidence Reporter for GreenLang predictions.

    This class provides a unified API for reporting prediction confidence
    and uncertainty, aggregating information from multiple sources
    (ensemble, conformal, calibration) into actionable insights.

    Key capabilities:
    - Unified confidence scoring
    - Prediction intervals
    - Uncertainty decomposition
    - Calibration quality
    - Human-readable reports
    - Provenance tracking

    Attributes:
        config: Reporter configuration
        ensemble: Optional ensemble predictor
        conformal: Optional conformal predictor
        calibrator: Optional calibrator

    Example:
        >>> reporter = ConfidenceReporter(
        ...     ensemble=ensemble_predictor,
        ...     conformal=conformal_predictor,
        ...     config=ConfidenceReporterConfig(interval_confidence=0.90)
        ... )
        >>> report = reporter.generate_report(X_test, predictions)
        >>> for pred in report.predictions[:5]:
        ...     print(f"Pred: {pred.prediction:.2f}, "
        ...           f"Confidence: {pred.confidence_level.value}")
    """

    def __init__(
        self,
        ensemble: Optional[Any] = None,
        conformal: Optional[Any] = None,
        calibrator: Optional[Any] = None,
        config: Optional[ConfidenceReporterConfig] = None
    ):
        """
        Initialize confidence reporter.

        Args:
            ensemble: EnsemblePredictor instance
            conformal: ConformalPredictor instance
            calibrator: Calibrator instance
            config: Reporter configuration
        """
        self.config = config or ConfidenceReporterConfig()
        self.ensemble = ensemble
        self.conformal = conformal
        self.calibrator = calibrator

        logger.info("ConfidenceReporter initialized")

    def _classify_confidence(self, score: float) -> ConfidenceLevel:
        """Classify confidence score into level."""
        thresholds = self.config.confidence_thresholds

        if score >= thresholds.get("very_high", 0.95):
            return ConfidenceLevel.VERY_HIGH
        elif score >= thresholds.get("high", 0.85):
            return ConfidenceLevel.HIGH
        elif score >= thresholds.get("medium", 0.70):
            return ConfidenceLevel.MEDIUM
        elif score >= thresholds.get("low", 0.50):
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _generate_recommendation(
        self,
        confidence_level: ConfidenceLevel,
        prediction: float
    ) -> str:
        """Generate action recommendation based on confidence."""
        recommendations = {
            ConfidenceLevel.VERY_HIGH: "Prediction is highly reliable. Safe for automated decision-making.",
            ConfidenceLevel.HIGH: "Prediction is reliable. May proceed with standard review.",
            ConfidenceLevel.MEDIUM: "Moderate confidence. Recommend human verification before action.",
            ConfidenceLevel.LOW: "Low confidence. Additional data or expert review recommended.",
            ConfidenceLevel.VERY_LOW: "Very low confidence. Do not use for decision-making without investigation."
        }
        return recommendations.get(confidence_level, "Unknown confidence level.")

    def _calculate_provenance(
        self,
        predictions: List[PredictionConfidence],
        overall_confidence: float
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        pred_sum = sum(p.prediction for p in predictions)
        conf_sum = sum(p.confidence_score for p in predictions)

        combined = f"{len(predictions)}|{pred_sum:.8f}|{conf_sum:.8f}|{overall_confidence:.8f}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _compute_confidence_score(
        self,
        ensemble_agreement: Optional[float] = None,
        interval_width: Optional[float] = None,
        calibrated_prob: Optional[float] = None,
        prediction_range: Optional[Tuple[float, float]] = None
    ) -> float:
        """
        Compute unified confidence score from multiple sources.

        Args:
            ensemble_agreement: Agreement between ensemble members
            interval_width: Width of prediction interval
            calibrated_prob: Calibrated probability
            prediction_range: Expected range for normalization

        Returns:
            Confidence score between 0 and 1
        """
        scores = []
        weights = []

        # Ensemble agreement (higher = more confident)
        if ensemble_agreement is not None:
            scores.append(ensemble_agreement)
            weights.append(0.4)

        # Interval width (narrower = more confident)
        if interval_width is not None and prediction_range is not None:
            range_size = prediction_range[1] - prediction_range[0]
            if range_size > 0:
                relative_width = interval_width / range_size
                width_score = max(0, 1 - relative_width)
                scores.append(width_score)
                weights.append(0.3)

        # Calibrated probability (closer to 0 or 1 = more confident for classification)
        if calibrated_prob is not None:
            prob_confidence = max(calibrated_prob, 1 - calibrated_prob)
            scores.append(prob_confidence)
            weights.append(0.3)

        if not scores:
            return 0.5  # Default moderate confidence

        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return float(np.clip(weighted_score, 0, 1))

    def generate_report(
        self,
        X: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None
    ) -> ConfidenceReport:
        """
        Generate comprehensive confidence report.

        Args:
            X: Input features
            predictions: Predictions (will compute if not provided)
            y_true: True values for coverage calculation

        Returns:
            ConfidenceReport with all confidence information

        Example:
            >>> report = reporter.generate_report(X_test)
            >>> print(f"High confidence: {report.high_confidence_ratio:.1%}")
        """
        logger.info(f"Generating confidence report for {len(X)} samples")

        prediction_confidences = []
        interval_widths = []
        ensemble_agreements = []

        # Get ensemble predictions and uncertainty
        ensemble_result = None
        if self.ensemble is not None and self.config.include_ensemble_agreement:
            try:
                ensemble_result = self.ensemble.predict_with_uncertainty(
                    X, confidence=self.config.interval_confidence
                )
            except Exception as e:
                logger.warning(f"Ensemble prediction failed: {e}")

        # Get conformal intervals
        conformal_result = None
        if self.conformal is not None and self.config.include_intervals:
            try:
                conformal_result = self.conformal.predict_interval(
                    X, confidence=self.config.interval_confidence
                )
            except Exception as e:
                logger.warning(f"Conformal prediction failed: {e}")

        # Estimate prediction range for normalization
        if predictions is not None:
            pred_range = (float(np.min(predictions)), float(np.max(predictions)))
        elif ensemble_result is not None:
            pred_range = (
                float(min(ensemble_result.lower_bounds)),
                float(max(ensemble_result.upper_bounds))
            )
        else:
            pred_range = (0, 1)

        # Generate per-prediction confidence
        n_samples = len(X)

        for i in range(n_samples):
            # Get prediction
            if predictions is not None:
                pred = float(predictions[i])
            elif ensemble_result is not None:
                pred = ensemble_result.predictions[i]
            else:
                pred = 0.0

            # Get ensemble info
            ensemble_std = None
            ensemble_agreement = None
            if ensemble_result is not None:
                ensemble_std = ensemble_result.uncertainties[i]
                ensemble_agreement = ensemble_result.model_agreement[i]
                ensemble_agreements.append(ensemble_agreement)

            # Get interval info
            lower_bound = None
            upper_bound = None
            interval_width = None

            if conformal_result is not None:
                lower_bound = conformal_result.intervals[i].lower
                upper_bound = conformal_result.intervals[i].upper
                interval_width = conformal_result.intervals[i].width
                interval_widths.append(interval_width)
            elif ensemble_result is not None:
                lower_bound = ensemble_result.lower_bounds[i]
                upper_bound = ensemble_result.upper_bounds[i]
                interval_width = upper_bound - lower_bound
                interval_widths.append(interval_width)

            # Compute confidence score
            confidence_score = self._compute_confidence_score(
                ensemble_agreement=ensemble_agreement,
                interval_width=interval_width,
                prediction_range=pred_range
            )

            # Classify confidence
            confidence_level = self._classify_confidence(confidence_score)

            # Generate recommendation
            recommendation = self._generate_recommendation(confidence_level, pred)

            # Decompose uncertainty sources
            uncertainty_sources = {}
            if ensemble_std is not None:
                uncertainty_sources["epistemic"] = float(ensemble_std)

            pred_confidence = PredictionConfidence(
                prediction=pred,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                interval_width=interval_width,
                ensemble_std=ensemble_std,
                ensemble_agreement=ensemble_agreement,
                uncertainty_sources=uncertainty_sources,
                recommendation=recommendation
            )

            prediction_confidences.append(pred_confidence)

        # Calculate overall statistics
        overall_confidence = float(np.mean([p.confidence_score for p in prediction_confidences]))

        # Confidence distribution
        confidence_distribution = {level.value: 0 for level in ConfidenceLevel}
        for p in prediction_confidences:
            confidence_distribution[p.confidence_level.value] += 1

        # High confidence ratio
        high_conf_levels = [ConfidenceLevel.VERY_HIGH.value, ConfidenceLevel.HIGH.value]
        high_confidence_count = sum(
            confidence_distribution.get(level, 0) for level in high_conf_levels
        )
        high_confidence_ratio = high_confidence_count / n_samples

        # Averages
        avg_interval_width = float(np.mean(interval_widths)) if interval_widths else None
        avg_ensemble_agreement = float(np.mean(ensemble_agreements)) if ensemble_agreements else None

        # Generate summary
        summary = self._generate_summary(
            overall_confidence,
            high_confidence_ratio,
            n_samples,
            avg_interval_width
        )

        # Generate recommendations
        recommendations = self._generate_overall_recommendations(
            prediction_confidences,
            high_confidence_ratio
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            prediction_confidences, overall_confidence
        )

        logger.info(
            f"Confidence report generated: "
            f"overall={overall_confidence:.3f}, high_conf={high_confidence_ratio:.1%}"
        )

        return ConfidenceReport(
            predictions=prediction_confidences,
            overall_confidence=overall_confidence,
            confidence_distribution=confidence_distribution,
            high_confidence_ratio=high_confidence_ratio,
            avg_interval_width=avg_interval_width,
            avg_ensemble_agreement=avg_ensemble_agreement,
            summary=summary,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow()
        )

    def _generate_summary(
        self,
        overall_confidence: float,
        high_confidence_ratio: float,
        n_samples: int,
        avg_interval_width: Optional[float]
    ) -> str:
        """Generate human-readable summary."""
        summary_parts = []

        # Overall assessment
        if overall_confidence >= 0.85:
            summary_parts.append(
                f"Predictions show high overall confidence ({overall_confidence:.1%})."
            )
        elif overall_confidence >= 0.70:
            summary_parts.append(
                f"Predictions show moderate confidence ({overall_confidence:.1%})."
            )
        else:
            summary_parts.append(
                f"Predictions show low confidence ({overall_confidence:.1%}). "
                "Careful review recommended."
            )

        # High confidence ratio
        summary_parts.append(
            f"{high_confidence_ratio:.1%} of {n_samples} predictions have high or very high confidence."
        )

        # Interval width
        if avg_interval_width is not None:
            summary_parts.append(
                f"Average prediction interval width: {avg_interval_width:.4f}."
            )

        return " ".join(summary_parts)

    def _generate_overall_recommendations(
        self,
        predictions: List[PredictionConfidence],
        high_confidence_ratio: float
    ) -> List[str]:
        """Generate overall recommendations."""
        recommendations = []

        if high_confidence_ratio < 0.5:
            recommendations.append(
                "Less than half of predictions have high confidence. "
                "Consider collecting more data or improving the model."
            )

        low_conf = [p for p in predictions if p.confidence_level in [
            ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW
        ]]
        if low_conf:
            recommendations.append(
                f"{len(low_conf)} predictions have low confidence and should be "
                "reviewed manually before use in decision-making."
            )

        if not recommendations:
            recommendations.append(
                "Overall prediction quality is good. "
                "Predictions can be used with standard monitoring."
            )

        return recommendations

    def format_for_api(
        self,
        report: ConfidenceReport
    ) -> Dict[str, Any]:
        """
        Format report for API response.

        Args:
            report: Confidence report

        Returns:
            API-friendly dictionary
        """
        return {
            "summary": report.summary,
            "overall_confidence": report.overall_confidence,
            "high_confidence_ratio": report.high_confidence_ratio,
            "n_predictions": len(report.predictions),
            "confidence_distribution": report.confidence_distribution,
            "recommendations": report.recommendations,
            "provenance_hash": report.provenance_hash,
            "timestamp": report.timestamp.isoformat(),
            "predictions": [
                {
                    "index": i,
                    "prediction": p.prediction,
                    "confidence_score": p.confidence_score,
                    "confidence_level": p.confidence_level.value,
                    "interval": [p.lower_bound, p.upper_bound] if p.lower_bound else None,
                    "recommendation": p.recommendation
                }
                for i, p in enumerate(report.predictions)
            ]
        }


# Unit test stubs
class TestConfidenceReporter:
    """Unit tests for ConfidenceReporter."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        reporter = ConfidenceReporter()
        assert reporter.config.interval_confidence == 0.95

    def test_classify_confidence(self):
        """Test confidence level classification."""
        reporter = ConfidenceReporter()

        assert reporter._classify_confidence(0.98) == ConfidenceLevel.VERY_HIGH
        assert reporter._classify_confidence(0.90) == ConfidenceLevel.HIGH
        assert reporter._classify_confidence(0.75) == ConfidenceLevel.MEDIUM
        assert reporter._classify_confidence(0.55) == ConfidenceLevel.LOW
        assert reporter._classify_confidence(0.30) == ConfidenceLevel.VERY_LOW

    def test_compute_confidence_score(self):
        """Test confidence score computation."""
        reporter = ConfidenceReporter()

        score = reporter._compute_confidence_score(
            ensemble_agreement=0.9,
            interval_width=0.1,
            prediction_range=(0, 1)
        )

        assert 0 <= score <= 1

    def test_generate_recommendation(self):
        """Test recommendation generation."""
        reporter = ConfidenceReporter()

        rec = reporter._generate_recommendation(ConfidenceLevel.VERY_HIGH, 1.0)
        assert "reliable" in rec.lower()

        rec = reporter._generate_recommendation(ConfidenceLevel.VERY_LOW, 1.0)
        assert "investigation" in rec.lower()

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        reporter = ConfidenceReporter()

        predictions = [
            PredictionConfidence(
                prediction=1.0,
                confidence_score=0.9,
                confidence_level=ConfidenceLevel.HIGH
            )
        ]

        hash1 = reporter._calculate_provenance(predictions, 0.9)
        hash2 = reporter._calculate_provenance(predictions, 0.9)

        assert hash1 == hash2
