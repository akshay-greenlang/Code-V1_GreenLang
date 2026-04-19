# -*- coding: utf-8 -*-
"""
ModelDriftDetector - Data and Concept Drift Detection

This module implements drift detection for ML models to identify when
model retraining is needed due to changes in data distribution or
model-target relationships.

Key Features:
    - Data drift detection using statistical tests
    - Concept drift detection via prediction monitoring
    - Retraining recommendations
    - Confidence degradation estimation

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    """Severity levels for drift detection."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(str, Enum):
    """Types of drift detected."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    LABEL_DRIFT = "label_drift"
    COVARIATE_SHIFT = "covariate_shift"


class ProvenanceRecord(BaseModel):
    """Provenance tracking for audit trails."""
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    calculation_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_id: str = Field(default="")
    input_hash: str = Field(default="")
    output_hash: str = Field(default="")
    computation_time_ms: float = Field(default=0.0)

    @classmethod
    def create(cls, calculation_type: str, inputs: Dict, outputs: Dict,
               model_id: str = "", computation_time_ms: float = 0.0) -> "ProvenanceRecord":
        return cls(
            calculation_type=calculation_type, model_id=model_id,
            input_hash=hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest(),
            output_hash=hashlib.sha256(json.dumps(outputs, sort_keys=True, default=str).encode()).hexdigest(),
            computation_time_ms=computation_time_ms
        )


class FeatureDrift(BaseModel):
    """Drift result for a single feature."""
    feature_name: str
    drift_detected: bool
    p_value: float = Field(ge=0.0, le=1.0)
    test_statistic: float
    drift_magnitude: float = Field(ge=0.0)
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float


class DriftResult(BaseModel):
    """Result from data drift detection."""
    detection_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    drift_detected: bool = Field(...)
    severity: DriftSeverity = Field(...)
    drift_type: DriftType = Field(default=DriftType.DATA_DRIFT)
    n_features_drifted: int = Field(default=0, ge=0)
    total_features: int = Field(default=0, ge=0)
    drift_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_p_value: float = Field(default=1.0, ge=0.0, le=1.0)
    feature_drifts: List[FeatureDrift] = Field(default_factory=list)
    top_drifted_features: List[str] = Field(default_factory=list)
    recommended_action: str = Field(default="")
    provenance: ProvenanceRecord


class ConceptDriftResult(BaseModel):
    """Result from concept drift detection."""
    detection_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    drift_detected: bool = Field(...)
    severity: DriftSeverity = Field(...)
    error_increase: float = Field(..., description="Increase in prediction error")
    error_increase_percent: float = Field(...)
    baseline_error: float = Field(...)
    current_error: float = Field(...)
    window_size: int = Field(default=100, ge=1)
    detection_method: str = Field(default="error_rate")
    recommended_action: str = Field(default="")
    provenance: ProvenanceRecord


class RetrainingRecommendation(BaseModel):
    """Recommendation for model retraining."""
    recommendation_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retrain_recommended: bool = Field(...)
    urgency: str = Field(default="low", description="low, medium, high, critical")
    reason: str = Field(default="")
    data_drift_score: float = Field(default=0.0, ge=0.0, le=1.0)
    concept_drift_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_degradation: float = Field(default=0.0, ge=0.0, le=1.0)
    recommended_actions: List[str] = Field(default_factory=list)
    estimated_improvement: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance: ProvenanceRecord


class ModelDriftDetector:
    """
    Drift detector for monitoring ML model health.

    Implements detection of:
    1. Data drift - changes in input feature distributions
    2. Concept drift - changes in relationship between features and target
    3. Confidence degradation over time since training

    Example:
        >>> detector = ModelDriftDetector()
        >>> drift = detector.detect_data_drift(current_data, reference_data)
        >>> if drift.drift_detected:
        ...     print(f"Drift detected in {drift.n_features_drifted} features")
    """

    # Default thresholds
    DEFAULT_P_VALUE_THRESHOLD = 0.05
    DEFAULT_DRIFT_FRACTION_THRESHOLD = 0.3
    DEFAULT_ERROR_INCREASE_THRESHOLD = 0.1

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        drift_fraction_threshold: float = 0.3,
        error_increase_threshold: float = 0.1
    ):
        """
        Initialize ModelDriftDetector.

        Args:
            p_value_threshold: P-value threshold for statistical tests
            drift_fraction_threshold: Fraction of features drifted to trigger alert
            error_increase_threshold: Error increase threshold for concept drift
        """
        self.p_value_threshold = p_value_threshold
        self.drift_fraction_threshold = drift_fraction_threshold
        self.error_increase_threshold = error_increase_threshold
        self._model_id = f"drift_detector_{uuid4().hex[:8]}"

        logger.info(f"ModelDriftDetector initialized: p_value={p_value_threshold}")

    def detect_data_drift(
        self,
        current: pd.DataFrame,
        reference: pd.DataFrame
    ) -> DriftResult:
        """
        Detect data drift between current and reference datasets.

        Uses Kolmogorov-Smirnov test for continuous features to detect
        distribution changes.

        Args:
            current: Current data window
            reference: Reference (training) data

        Returns:
            DriftResult with drift analysis
        """
        start_time = time.time()

        # Find common numeric columns
        numeric_cols = [c for c in current.columns
                       if c in reference.columns and
                       pd.api.types.is_numeric_dtype(current[c])]

        if not numeric_cols:
            return self._no_drift_result("No numeric columns to compare")

        feature_drifts = []
        drifted_features = []

        for col in numeric_cols:
            curr_data = current[col].dropna().values
            ref_data = reference[col].dropna().values

            if len(curr_data) < 10 or len(ref_data) < 10:
                continue

            # Perform KS test
            if SCIPY_AVAILABLE:
                statistic, p_value = stats.ks_2samp(curr_data, ref_data)
            else:
                # Fallback: simple mean comparison
                statistic = abs(np.mean(curr_data) - np.mean(ref_data)) / (np.std(ref_data) + 1e-8)
                p_value = 1.0 if statistic < 0.5 else 0.01

            drift_detected = p_value < self.p_value_threshold
            drift_magnitude = abs(np.mean(curr_data) - np.mean(ref_data)) / (np.std(ref_data) + 1e-8)

            feature_drift = FeatureDrift(
                feature_name=col,
                drift_detected=drift_detected,
                p_value=float(p_value),
                test_statistic=float(statistic),
                drift_magnitude=float(drift_magnitude),
                reference_mean=float(np.mean(ref_data)),
                current_mean=float(np.mean(curr_data)),
                reference_std=float(np.std(ref_data)),
                current_std=float(np.std(curr_data))
            )
            feature_drifts.append(feature_drift)

            if drift_detected:
                drifted_features.append(col)

        total_features = len(feature_drifts)
        n_drifted = len(drifted_features)
        drift_fraction = n_drifted / total_features if total_features > 0 else 0.0

        # Overall drift detection
        overall_drift = drift_fraction >= self.drift_fraction_threshold

        # Determine severity
        if not overall_drift:
            severity = DriftSeverity.NONE
        elif drift_fraction < 0.2:
            severity = DriftSeverity.LOW
        elif drift_fraction < 0.4:
            severity = DriftSeverity.MEDIUM
        elif drift_fraction < 0.6:
            severity = DriftSeverity.HIGH
        else:
            severity = DriftSeverity.CRITICAL

        # Combined p-value (Fisher's method)
        if SCIPY_AVAILABLE and feature_drifts:
            p_values = [fd.p_value for fd in feature_drifts]
            chi2_stat = -2 * sum(np.log(max(p, 1e-10)) for p in p_values)
            overall_p = float(1 - stats.chi2.cdf(chi2_stat, 2 * len(p_values)))
        else:
            overall_p = min((fd.p_value for fd in feature_drifts), default=1.0)

        # Top drifted features
        sorted_drifts = sorted(feature_drifts, key=lambda x: x.drift_magnitude, reverse=True)
        top_drifted = [fd.feature_name for fd in sorted_drifts[:5] if fd.drift_detected]

        # Recommended action
        if severity == DriftSeverity.CRITICAL:
            action = "IMMEDIATE: Investigate data pipeline and retrain model"
        elif severity == DriftSeverity.HIGH:
            action = "Schedule model retraining within 24 hours"
        elif severity == DriftSeverity.MEDIUM:
            action = "Monitor closely and plan retraining"
        elif severity == DriftSeverity.LOW:
            action = "Continue monitoring"
        else:
            action = "No action required"

        computation_time_ms = (time.time() - start_time) * 1000

        return DriftResult(
            drift_detected=overall_drift,
            severity=severity,
            drift_type=DriftType.DATA_DRIFT,
            n_features_drifted=n_drifted,
            total_features=total_features,
            drift_fraction=drift_fraction,
            overall_p_value=overall_p,
            feature_drifts=feature_drifts,
            top_drifted_features=top_drifted,
            recommended_action=action,
            provenance=ProvenanceRecord.create(
                "data_drift_detection",
                {"n_current": len(current), "n_reference": len(reference)},
                {"drift_detected": overall_drift, "severity": severity.value},
                self._model_id, computation_time_ms
            )
        )

    def detect_concept_drift(
        self,
        predictions: List[float],
        actuals: List[float],
        baseline_error: Optional[float] = None,
        window_size: int = 100
    ) -> ConceptDriftResult:
        """
        Detect concept drift by monitoring prediction error.

        Concept drift occurs when the relationship between features and
        target changes, even if feature distributions remain stable.

        Args:
            predictions: Recent model predictions
            actuals: Actual target values
            baseline_error: Baseline error rate (from validation)
            window_size: Window size for error calculation

        Returns:
            ConceptDriftResult with drift analysis
        """
        start_time = time.time()

        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        if len(predictions) < 10:
            return self._no_concept_drift_result("Insufficient data points")

        preds = np.array(predictions[-window_size:])
        acts = np.array(actuals[-window_size:])

        # Calculate current error (MAE)
        current_error = float(np.mean(np.abs(preds - acts)))

        # Use baseline if provided, otherwise estimate from first half
        if baseline_error is None:
            if len(predictions) > 2 * window_size:
                early_preds = np.array(predictions[:window_size])
                early_acts = np.array(actuals[:window_size])
                baseline_error = float(np.mean(np.abs(early_preds - early_acts)))
            else:
                baseline_error = current_error * 0.8  # Assume 20% degradation is drift

        # Calculate error increase
        error_increase = current_error - baseline_error
        if baseline_error > 0:
            error_increase_percent = 100.0 * error_increase / baseline_error
        else:
            error_increase_percent = 100.0 if error_increase > 0 else 0.0

        # Detect drift
        drift_detected = error_increase / (baseline_error + 1e-8) > self.error_increase_threshold

        # Determine severity
        relative_increase = error_increase / (baseline_error + 1e-8)
        if not drift_detected:
            severity = DriftSeverity.NONE
        elif relative_increase < 0.2:
            severity = DriftSeverity.LOW
        elif relative_increase < 0.4:
            severity = DriftSeverity.MEDIUM
        elif relative_increase < 0.6:
            severity = DriftSeverity.HIGH
        else:
            severity = DriftSeverity.CRITICAL

        # Recommended action
        if severity == DriftSeverity.CRITICAL:
            action = "IMMEDIATE: Model performance degraded significantly. Retrain urgently."
        elif severity == DriftSeverity.HIGH:
            action = "Schedule immediate model retraining"
        elif severity == DriftSeverity.MEDIUM:
            action = "Plan model retraining in next maintenance window"
        elif severity == DriftSeverity.LOW:
            action = "Monitor prediction quality closely"
        else:
            action = "No action required"

        computation_time_ms = (time.time() - start_time) * 1000

        return ConceptDriftResult(
            drift_detected=drift_detected,
            severity=severity,
            error_increase=error_increase,
            error_increase_percent=error_increase_percent,
            baseline_error=baseline_error,
            current_error=current_error,
            window_size=min(window_size, len(predictions)),
            detection_method="mae_comparison",
            recommended_action=action,
            provenance=ProvenanceRecord.create(
                "concept_drift_detection",
                {"n_predictions": len(predictions), "window_size": window_size},
                {"drift_detected": drift_detected, "error_increase": error_increase},
                self._model_id, computation_time_ms
            )
        )

    def recommend_retraining(
        self,
        drift_metrics: Dict[str, Any]
    ) -> RetrainingRecommendation:
        """
        Generate retraining recommendation based on drift metrics.

        Args:
            drift_metrics: Dictionary with drift detection results
                - data_drift_score: 0-1 score for data drift
                - concept_drift_score: 0-1 score for concept drift
                - time_since_training_days: Days since last training
                - prediction_volume: Number of predictions since training

        Returns:
            RetrainingRecommendation with analysis and actions
        """
        start_time = time.time()

        data_drift_score = drift_metrics.get('data_drift_score', 0.0)
        concept_drift_score = drift_metrics.get('concept_drift_score', 0.0)
        time_since_training = drift_metrics.get('time_since_training_days', 0)
        prediction_volume = drift_metrics.get('prediction_volume', 0)

        # Calculate confidence degradation
        conf_degradation = self.compute_confidence_degradation(time_since_training)

        # Combined drift score
        combined_score = max(data_drift_score, concept_drift_score, conf_degradation)

        # Determine if retraining is recommended
        retrain_recommended = combined_score > 0.3

        # Determine urgency
        if combined_score > 0.7:
            urgency = "critical"
        elif combined_score > 0.5:
            urgency = "high"
        elif combined_score > 0.3:
            urgency = "medium"
        else:
            urgency = "low"

        # Generate reason
        reasons = []
        if data_drift_score > 0.3:
            reasons.append(f"Data drift detected (score: {data_drift_score:.2f})")
        if concept_drift_score > 0.3:
            reasons.append(f"Concept drift detected (score: {concept_drift_score:.2f})")
        if conf_degradation > 0.3:
            reasons.append(f"Model age degradation (score: {conf_degradation:.2f})")
        if time_since_training > 90:
            reasons.append(f"Model trained {time_since_training} days ago")

        reason = "; ".join(reasons) if reasons else "Model performing within acceptable limits"

        # Recommended actions
        actions = []
        if retrain_recommended:
            if urgency == "critical":
                actions.append("IMMEDIATE: Initiate emergency retraining")
                actions.append("Consider falling back to physics-based model")
            elif urgency == "high":
                actions.append("Schedule retraining within 24 hours")
                actions.append("Increase monitoring frequency")
            else:
                actions.append("Plan retraining in next maintenance window")
                actions.append("Collect recent data for training")

            if data_drift_score > 0.3:
                actions.append("Investigate data pipeline for anomalies")
            if concept_drift_score > 0.3:
                actions.append("Verify target variable definition is consistent")
        else:
            actions.append("Continue regular monitoring")
            actions.append("No immediate action required")

        # Estimated improvement
        if retrain_recommended:
            estimated_improvement = min(0.5, combined_score * 0.7)
        else:
            estimated_improvement = 0.0

        computation_time_ms = (time.time() - start_time) * 1000

        return RetrainingRecommendation(
            retrain_recommended=retrain_recommended,
            urgency=urgency,
            reason=reason,
            data_drift_score=data_drift_score,
            concept_drift_score=concept_drift_score,
            confidence_degradation=conf_degradation,
            recommended_actions=actions,
            estimated_improvement=estimated_improvement,
            provenance=ProvenanceRecord.create(
                "retraining_recommendation",
                drift_metrics,
                {"recommended": retrain_recommended, "urgency": urgency},
                self._model_id, computation_time_ms
            )
        )

    def compute_confidence_degradation(
        self,
        time_since_train: float,
        half_life_days: float = 90.0
    ) -> float:
        """
        Compute confidence degradation based on time since training.

        Uses exponential decay model where confidence degrades over time.

        Args:
            time_since_train: Days since model was trained
            half_life_days: Days for confidence to degrade by 50%

        Returns:
            Degradation score [0, 1] where 0=no degradation, 1=full degradation
        """
        if time_since_train <= 0:
            return 0.0

        # Exponential decay: degradation = 1 - exp(-t/tau)
        # where tau = half_life / ln(2)
        tau = half_life_days / np.log(2)
        degradation = 1.0 - np.exp(-time_since_train / tau)

        return float(min(1.0, max(0.0, degradation)))

    def _no_drift_result(self, message: str) -> DriftResult:
        """Return a no-drift result."""
        return DriftResult(
            drift_detected=False,
            severity=DriftSeverity.NONE,
            n_features_drifted=0,
            total_features=0,
            drift_fraction=0.0,
            overall_p_value=1.0,
            recommended_action=message,
            provenance=ProvenanceRecord.create("data_drift_detection", {}, {}, self._model_id, 0)
        )

    def _no_concept_drift_result(self, message: str) -> ConceptDriftResult:
        """Return a no-concept-drift result."""
        return ConceptDriftResult(
            drift_detected=False,
            severity=DriftSeverity.NONE,
            error_increase=0.0,
            error_increase_percent=0.0,
            baseline_error=0.0,
            current_error=0.0,
            recommended_action=message,
            provenance=ProvenanceRecord.create("concept_drift_detection", {}, {}, self._model_id, 0)
        )

    @property
    def model_id(self) -> str:
        """Get model identifier."""
        return self._model_id
