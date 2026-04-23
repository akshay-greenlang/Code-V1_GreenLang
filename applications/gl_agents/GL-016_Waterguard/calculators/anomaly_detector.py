# -*- coding: utf-8 -*-
"""
GL-016 Waterguard Anomaly Detector - Multi-variate Anomaly Detection

This module provides multi-variate anomaly detection for water chemistry monitoring.
Uses statistical methods and Isolation Forest for detecting analyzer drift,
reagent/sample-flow faults, and multivariate anomalies.

ALL CALCULATIONS ARE DETERMINISTIC - NO GENERATIVE AI FOR NUMERIC DECISIONS.

Example:
    >>> detector = WaterguardAnomalyDetector(config)
    >>> detector.fit_baseline(historical_data)
    >>> result = detector.detect_anomalies(sensor_data)
    >>> if result.detected:
    ...     print(f"Anomaly: {result.anomaly_type}, Severity: {result.severity}")

Author: GreenLang Waterguard Team
Version: 1.0.0
Agent: GL-016_Waterguard
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""
    SLOW_BIAS = "SLOW_BIAS"
    STEP_CHANGE = "STEP_CHANGE"
    REAGENT_FAULT = "REAGENT_FAULT"
    SAMPLE_FLOW_FAULT = "SAMPLE_FLOW_FAULT"
    MULTIVARIATE_OUTLIER = "MULTIVARIATE_OUTLIER"
    CHANGE_POINT = "CHANGE_POINT"
    FLATLINE = "FLATLINE"
    SPIKE = "SPIKE"
    CORRELATION_BREAK = "CORRELATION_BREAK"


class AnomalySeverity(str, Enum):
    """Severity levels for detected anomalies."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class AnomalyDetectorConfig:
    """Configuration for WaterguardAnomalyDetector."""

    # Isolation Forest parameters
    contamination: float = 0.05
    n_estimators: int = 100
    random_state: int = 42

    # Change point detection parameters
    change_point_window: int = 20
    change_point_threshold: float = 3.0

    # Step change detection
    step_change_threshold: float = 2.5

    # Flatline detection
    flatline_tolerance: float = 0.001
    flatline_duration: int = 10

    # Reagent flow thresholds (mL/min)
    reagent_flow_min: float = 0.5
    reagent_flow_max: float = 10.0

    # Sample flow thresholds (mL/min)
    sample_flow_min: float = 50.0
    sample_flow_max: float = 500.0

    # Confidence thresholds
    min_confidence: float = 0.7

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.contamination <= 0.5:
            raise ValueError("contamination must be between 0.0 and 0.5")
        if not 10 <= self.n_estimators <= 500:
            raise ValueError("n_estimators must be between 10 and 500")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")


@dataclass
class SensorData:
    """Collection of sensor readings for analysis."""

    readings: Dict[str, List[float]]
    timestamps: List[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate sensor data."""
        if self.readings and self.timestamps:
            first_tag = list(self.readings.keys())[0]
            if len(self.timestamps) != len(self.readings[first_tag]):
                raise ValueError("Timestamps must match readings length")


@dataclass
class AnomalyResult:
    """Result from anomaly detection."""

    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detected: bool = False
    anomaly_type: Optional[AnomalyType] = None
    severity: Optional[AnomalySeverity] = None
    confidence: float = 0.0
    affected_tags: List[str] = field(default_factory=list)
    detection_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    anomaly_start_time: Optional[datetime] = None
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    provenance_hash: str = ""


class WaterguardAnomalyDetector:
    """
    Multi-variate anomaly detection for water chemistry monitoring.

    This detector uses ONLY statistical methods (no generative AI) for
    anomaly detection, ensuring zero-hallucination compliance per GreenLang
    standards.

    Detection Methods:
        - Isolation Forest: Multivariate outlier detection
        - CUSUM: Change point detection
        - Z-score: Step change detection
        - Variance: Flatline detection
        - Threshold: Reagent/sample flow faults

    Attributes:
        config: Detector configuration
        _isolation_forest: Trained Isolation Forest model (lazy loaded)
        _baseline_stats: Baseline statistics for each sensor

    Example:
        >>> config = AnomalyDetectorConfig(contamination=0.05)
        >>> detector = WaterguardAnomalyDetector(config)
        >>> detector.fit_baseline(historical_data)
        >>> result = detector.detect_anomalies(current_data)
    """

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        """
        Initialize WaterguardAnomalyDetector.

        Args:
            config: Detector configuration. Uses defaults if None.
        """
        self.config = config or AnomalyDetectorConfig()
        self._isolation_forest = None
        self._baseline_stats: Dict[str, Dict[str, float]] = {}
        self._is_fitted = False
        self._feature_tags: List[str] = []

        logger.info(
            f"WaterguardAnomalyDetector initialized with contamination="
            f"{self.config.contamination}"
        )

    def fit_baseline(
        self,
        historical_data: SensorData,
        exclude_outliers: bool = True
    ) -> None:
        """
        Fit baseline statistics from historical data.

        Args:
            historical_data: Historical sensor data for baseline
            exclude_outliers: Whether to exclude outliers when computing baseline

        Raises:
            ValueError: If historical data is insufficient
        """
        start_time = datetime.now()
        logger.info("Fitting baseline statistics from historical data")

        if not historical_data.readings:
            raise ValueError("Historical data cannot be empty")

        for tag_id, values in historical_data.readings.items():
            if len(values) < 10:
                logger.warning(f"Insufficient data for {tag_id}: {len(values)} samples")
                continue

            values_arr = np.array(values, dtype=np.float64)

            # Optionally exclude outliers using IQR
            if exclude_outliers:
                q1, q3 = np.percentile(values_arr, [25, 75])
                iqr = q3 - q1
                mask = (values_arr >= q1 - 1.5 * iqr) & (values_arr <= q3 + 1.5 * iqr)
                clean_values = values_arr[mask]
                if len(clean_values) < 5:
                    clean_values = values_arr
            else:
                clean_values = values_arr

            # Compute baseline statistics
            self._baseline_stats[tag_id] = {
                "mean": float(np.mean(clean_values)),
                "std": float(np.std(clean_values)),
                "median": float(np.median(clean_values)),
                "min": float(np.min(clean_values)),
                "max": float(np.max(clean_values)),
                "q1": float(np.percentile(clean_values, 25)),
                "q3": float(np.percentile(clean_values, 75)),
                "sample_count": len(clean_values),
            }

        # Fit Isolation Forest if enough multi-variate data
        if len(historical_data.readings) >= 2:
            self._fit_isolation_forest(historical_data)

        self._is_fitted = True
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            f"Baseline fitting complete: {len(self._baseline_stats)} sensors, "
            f"{duration_ms:.2f}ms"
        )

    def _fit_isolation_forest(self, data: SensorData) -> None:
        """Fit Isolation Forest model for multivariate detection."""
        try:
            from sklearn.ensemble import IsolationForest

            # Build feature matrix
            self._feature_tags = sorted(data.readings.keys())
            n_samples = len(data.timestamps)
            X = np.zeros((n_samples, len(self._feature_tags)))

            for i, tag in enumerate(self._feature_tags):
                X[:, i] = data.readings[tag]

            # Fit model
            self._isolation_forest = IsolationForest(
                contamination=self.config.contamination,
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            self._isolation_forest.fit(X)
            logger.info("Isolation Forest fitted successfully")

        except ImportError:
            logger.warning("sklearn not available, skipping Isolation Forest")
            self._isolation_forest = None

    def detect_anomalies(
        self,
        sensor_data: SensorData,
        check_all: bool = True
    ) -> List[AnomalyResult]:
        """
        Detect anomalies in sensor data.

        Args:
            sensor_data: Current sensor readings
            check_all: Whether to run all detection methods

        Returns:
            List of detected anomalies (may be empty)

        Raises:
            ValueError: If detector not fitted
        """
        if not self._is_fitted:
            raise ValueError("Detector must be fitted before detection")

        start_time = datetime.now()
        results: List[AnomalyResult] = []

        if check_all:
            # 1. Check for step changes
            results.extend(self._detect_step_changes(sensor_data))

            # 2. Check for flatlines
            results.extend(self._detect_flatlines(sensor_data))

            # 3. Check for change points
            results.extend(self._detect_change_points(sensor_data))

            # 4. Check multivariate anomalies
            if self._isolation_forest is not None:
                results.extend(self._detect_multivariate_anomalies(sensor_data))

            # 5. Check reagent flow
            results.extend(self._detect_reagent_faults(sensor_data))

            # 6. Check sample flow
            results.extend(self._detect_sample_flow_faults(sensor_data))

        # Filter by confidence threshold
        results = [r for r in results if r.confidence >= self.config.min_confidence]

        # Add provenance hashes
        for result in results:
            result.provenance_hash = self._calculate_provenance(result, sensor_data)

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            f"Anomaly detection complete: {len(results)} anomalies detected, "
            f"{duration_ms:.2f}ms"
        )

        return results

    def _detect_step_changes(self, data: SensorData) -> List[AnomalyResult]:
        """Detect step changes using z-score method."""
        results = []

        for tag_id, values in data.readings.items():
            if tag_id not in self._baseline_stats:
                continue

            baseline = self._baseline_stats[tag_id]
            if baseline["std"] < 1e-10:
                continue

            values_arr = np.array(values)
            if len(values_arr) < 2:
                continue

            # Calculate z-scores relative to baseline
            z_scores = (values_arr - baseline["mean"]) / baseline["std"]

            # Find significant deviations
            threshold = self.config.step_change_threshold
            step_indices = np.where(np.abs(z_scores) > threshold)[0]

            if len(step_indices) >= 3:
                first_idx = step_indices[0]
                max_z = float(np.max(np.abs(z_scores[step_indices])))
                confidence = min(1.0, max_z / (threshold * 2))
                severity = self._determine_severity(max_z, threshold)

                results.append(AnomalyResult(
                    detected=True,
                    anomaly_type=AnomalyType.STEP_CHANGE,
                    severity=severity,
                    confidence=confidence,
                    affected_tags=[tag_id],
                    anomaly_start_time=data.timestamps[first_idx] if data.timestamps else None,
                    description=f"Step change detected in {tag_id}: z-score={max_z:.2f}",
                    metrics={
                        "max_z_score": max_z,
                        "baseline_mean": baseline["mean"],
                        "baseline_std": baseline["std"],
                        "current_mean": float(np.mean(values_arr[step_indices])),
                    }
                ))

        return results

    def _detect_flatlines(self, data: SensorData) -> List[AnomalyResult]:
        """Detect flatline conditions (stuck sensors)."""
        results = []

        for tag_id, values in data.readings.items():
            values_arr = np.array(values)
            if len(values_arr) < self.config.flatline_duration:
                continue

            window = self.config.flatline_duration
            for i in range(len(values_arr) - window + 1):
                window_values = values_arr[i:i + window]
                variance = np.var(window_values)

                if variance < self.config.flatline_tolerance:
                    confidence = 1.0 - (variance / self.config.flatline_tolerance)

                    results.append(AnomalyResult(
                        detected=True,
                        anomaly_type=AnomalyType.FLATLINE,
                        severity=AnomalySeverity.HIGH,
                        confidence=confidence,
                        affected_tags=[tag_id],
                        anomaly_start_time=data.timestamps[i] if data.timestamps else None,
                        description=f"Flatline detected in {tag_id}: variance={variance:.6f}",
                        metrics={
                            "variance": float(variance),
                            "flatline_value": float(np.mean(window_values)),
                            "duration_samples": window,
                        }
                    ))
                    break

        return results

    def _detect_change_points(self, data: SensorData) -> List[AnomalyResult]:
        """Detect change points using CUSUM algorithm."""
        results = []

        for tag_id, values in data.readings.items():
            if tag_id not in self._baseline_stats:
                continue

            values_arr = np.array(values)
            if len(values_arr) < self.config.change_point_window * 2:
                continue

            baseline = self._baseline_stats[tag_id]
            target = baseline["mean"]
            sigma = max(baseline["std"], 1e-10)

            # CUSUM calculation
            cusum_pos = np.zeros(len(values_arr))
            cusum_neg = np.zeros(len(values_arr))
            k = 0.5

            for i in range(1, len(values_arr)):
                z = (values_arr[i] - target) / sigma
                cusum_pos[i] = max(0, cusum_pos[i-1] + z - k)
                cusum_neg[i] = max(0, cusum_neg[i-1] - z - k)

            h = self.config.change_point_threshold
            change_idx = np.where((cusum_pos > h) | (cusum_neg > h))[0]

            if len(change_idx) > 0:
                first_change = change_idx[0]
                max_cusum = max(np.max(cusum_pos), np.max(cusum_neg))
                confidence = min(1.0, max_cusum / (h * 2))

                results.append(AnomalyResult(
                    detected=True,
                    anomaly_type=AnomalyType.CHANGE_POINT,
                    severity=self._determine_severity(max_cusum, h),
                    confidence=confidence,
                    affected_tags=[tag_id],
                    anomaly_start_time=data.timestamps[first_change] if data.timestamps else None,
                    description=f"Change point detected in {tag_id}: CUSUM={max_cusum:.2f}",
                    metrics={
                        "cusum_value": float(max_cusum),
                        "threshold": h,
                        "change_point_index": int(first_change),
                    }
                ))

        return results

    def _detect_multivariate_anomalies(self, data: SensorData) -> List[AnomalyResult]:
        """Detect multivariate anomalies using Isolation Forest."""
        results = []

        if self._isolation_forest is None:
            return results

        try:
            n_samples = len(data.timestamps)
            X = np.zeros((n_samples, len(self._feature_tags)))

            for i, tag in enumerate(self._feature_tags):
                if tag in data.readings:
                    X[:, i] = data.readings[tag]

            predictions = self._isolation_forest.predict(X)
            scores = self._isolation_forest.decision_function(X)

            anomaly_indices = np.where(predictions == -1)[0]

            if len(anomaly_indices) > 0:
                anomaly_scores = scores[anomaly_indices]
                avg_score = np.mean(np.abs(anomaly_scores))
                confidence = min(1.0, avg_score * 2)

                results.append(AnomalyResult(
                    detected=True,
                    anomaly_type=AnomalyType.MULTIVARIATE_OUTLIER,
                    severity=self._determine_severity_from_count(len(anomaly_indices), n_samples),
                    confidence=confidence,
                    affected_tags=self._feature_tags,
                    anomaly_start_time=data.timestamps[anomaly_indices[0]] if data.timestamps else None,
                    description=f"Multivariate anomaly detected: {len(anomaly_indices)} outliers",
                    metrics={
                        "outlier_count": len(anomaly_indices),
                        "total_samples": n_samples,
                        "outlier_ratio": len(anomaly_indices) / n_samples,
                        "avg_anomaly_score": float(avg_score),
                    }
                ))

        except Exception as e:
            logger.error(f"Error in multivariate detection: {e}")

        return results

    def _detect_reagent_faults(self, data: SensorData) -> List[AnomalyResult]:
        """Detect reagent flow faults."""
        results = []

        reagent_tags = [t for t in data.readings.keys()
                        if "reagent" in t.lower() or "chemical" in t.lower()]

        for tag_id in reagent_tags:
            values = np.array(data.readings[tag_id])
            out_of_range = (values < self.config.reagent_flow_min) | \
                           (values > self.config.reagent_flow_max)

            if np.any(out_of_range):
                fault_indices = np.where(out_of_range)[0]
                fault_values = values[out_of_range]
                confidence = min(1.0, len(fault_indices) / len(values) * 2)

                results.append(AnomalyResult(
                    detected=True,
                    anomaly_type=AnomalyType.REAGENT_FAULT,
                    severity=AnomalySeverity.HIGH,
                    confidence=confidence,
                    affected_tags=[tag_id],
                    anomaly_start_time=data.timestamps[fault_indices[0]] if data.timestamps else None,
                    description=f"Reagent flow fault in {tag_id}",
                    metrics={
                        "fault_count": len(fault_indices),
                        "min_value": float(np.min(fault_values)),
                        "max_value": float(np.max(fault_values)),
                        "threshold_min": self.config.reagent_flow_min,
                        "threshold_max": self.config.reagent_flow_max,
                    }
                ))

        return results

    def _detect_sample_flow_faults(self, data: SensorData) -> List[AnomalyResult]:
        """Detect sample flow faults."""
        results = []

        sample_tags = [t for t in data.readings.keys()
                       if "sample" in t.lower() and "flow" in t.lower()]

        for tag_id in sample_tags:
            values = np.array(data.readings[tag_id])
            out_of_range = (values < self.config.sample_flow_min) | \
                           (values > self.config.sample_flow_max)

            if np.any(out_of_range):
                fault_indices = np.where(out_of_range)[0]
                fault_values = values[out_of_range]
                confidence = min(1.0, len(fault_indices) / len(values) * 2)

                results.append(AnomalyResult(
                    detected=True,
                    anomaly_type=AnomalyType.SAMPLE_FLOW_FAULT,
                    severity=AnomalySeverity.HIGH,
                    confidence=confidence,
                    affected_tags=[tag_id],
                    anomaly_start_time=data.timestamps[fault_indices[0]] if data.timestamps else None,
                    description=f"Sample flow fault in {tag_id}",
                    metrics={
                        "fault_count": len(fault_indices),
                        "min_value": float(np.min(fault_values)),
                        "max_value": float(np.max(fault_values)),
                        "threshold_min": self.config.sample_flow_min,
                        "threshold_max": self.config.sample_flow_max,
                    }
                ))

        return results

    def _determine_severity(self, score: float, threshold: float) -> AnomalySeverity:
        """Determine severity based on score relative to threshold."""
        ratio = score / threshold
        if ratio >= 4.0:
            return AnomalySeverity.CRITICAL
        elif ratio >= 2.5:
            return AnomalySeverity.HIGH
        elif ratio >= 1.5:
            return AnomalySeverity.MEDIUM
        return AnomalySeverity.LOW

    def _determine_severity_from_count(self, count: int, total: int) -> AnomalySeverity:
        """Determine severity based on anomaly count ratio."""
        ratio = count / total
        if ratio >= 0.3:
            return AnomalySeverity.CRITICAL
        elif ratio >= 0.15:
            return AnomalySeverity.HIGH
        elif ratio >= 0.05:
            return AnomalySeverity.MEDIUM
        return AnomalySeverity.LOW

    def _calculate_provenance(self, result: AnomalyResult, data: SensorData) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{result.result_id}"
            f"{result.anomaly_type}"
            f"{result.detection_time.isoformat()}"
            f"{str(sorted(data.readings.keys()))}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def get_baseline_stats(self, tag_id: str) -> Optional[Dict[str, float]]:
        """Get baseline statistics for a sensor."""
        return self._baseline_stats.get(tag_id)

    def is_fitted(self) -> bool:
        """Check if detector has been fitted."""
        return self._is_fitted
