# -*- coding: utf-8 -*-
"""
GL-016 Waterguard Drift Detector - Sensor Drift Detection

This module provides sensor drift detection for water chemistry analyzers.
Detects slow bias, step changes, flatlines, and reagent flow anomalies.

ALL CALCULATIONS ARE DETERMINISTIC - NO GENERATIVE AI FOR NUMERIC DECISIONS.

Example:
    >>> detector = SensorDriftDetector(config)
    >>> result = detector.detect_slow_drift(sensor_history, expected_baseline=7.0)
    >>> if result.detected:
    ...     print(f"Drift: {result.drift_type}, Magnitude: {result.magnitude}")

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
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    """Types of sensor drift that can be detected."""
    SLOW_BIAS = "SLOW_BIAS"
    STEP_CHANGE = "STEP_CHANGE"
    FLATLINE = "FLATLINE"
    REAGENT_FLOW_LOW = "REAGENT_FLOW_LOW"
    REAGENT_FLOW_HIGH = "REAGENT_FLOW_HIGH"
    REAGENT_FLOW_ERRATIC = "REAGENT_FLOW_ERRATIC"
    INCREASING_TREND = "INCREASING_TREND"
    DECREASING_TREND = "DECREASING_TREND"
    OSCILLATING = "OSCILLATING"


@dataclass
class DriftDetectorConfig:
    """Configuration for SensorDriftDetector."""

    # Slow drift detection
    slow_drift_window: int = 100
    slow_drift_threshold_pct: float = 5.0
    trend_min_r_squared: float = 0.6

    # Step change detection
    step_change_sigma: float = 3.0
    step_change_window: int = 10

    # Flatline detection
    flatline_tolerance: float = 0.001
    flatline_min_duration: int = 10
    flatline_max_variance: float = 1e-6

    # Reagent flow monitoring
    reagent_flow_min: float = 0.5
    reagent_flow_max: float = 10.0
    reagent_flow_cv_threshold: float = 0.3

    # Confidence thresholds
    min_confidence: float = 0.7

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if self.slow_drift_window < 10:
            raise ValueError("slow_drift_window must be at least 10")


@dataclass
class DriftResult:
    """Result from drift detection."""

    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detected: bool = False
    drift_type: Optional[DriftType] = None
    magnitude: float = 0.0
    time_detected: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    drift_start_time: Optional[datetime] = None
    confidence: float = 0.0
    affected_tag: str = ""
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "result_id": self.result_id,
            "detected": self.detected,
            "drift_type": self.drift_type.value if self.drift_type else None,
            "magnitude": self.magnitude,
            "time_detected": self.time_detected.isoformat(),
            "drift_start_time": self.drift_start_time.isoformat() if self.drift_start_time else None,
            "confidence": self.confidence,
            "affected_tag": self.affected_tag,
            "description": self.description,
            "metrics": self.metrics,
            "provenance_hash": self.provenance_hash,
        }


class SensorDriftDetector:
    """
    Sensor drift detection for water chemistry analyzers.

    This detector uses ONLY statistical methods for drift detection,
    ensuring zero-hallucination compliance per GreenLang standards.

    Detection Methods:
        - Linear regression: Slow drift/bias detection
        - Z-score: Step change detection
        - Variance analysis: Flatline detection
        - Threshold/CV: Reagent flow monitoring

    Attributes:
        config: Detector configuration

    Example:
        >>> config = DriftDetectorConfig(slow_drift_threshold_pct=5.0)
        >>> detector = SensorDriftDetector(config)
        >>> result = detector.detect_slow_drift(values, baseline=7.0)
    """

    def __init__(self, config: Optional[DriftDetectorConfig] = None):
        """
        Initialize SensorDriftDetector.

        Args:
            config: Detector configuration. Uses defaults if None.
        """
        self.config = config or DriftDetectorConfig()
        logger.info(
            f"SensorDriftDetector initialized with drift_threshold="
            f"{self.config.slow_drift_threshold_pct}%"
        )

    def detect_slow_drift(
        self,
        sensor_history: List[float],
        expected_baseline: float,
        timestamps: Optional[List[datetime]] = None,
        tag_id: str = ""
    ) -> DriftResult:
        """
        Detect slow bias/drift in sensor readings.

        Uses linear regression to detect gradual drift from baseline.
        Zero-hallucination: Uses deterministic statistical regression.

        Args:
            sensor_history: Historical sensor values (oldest to newest)
            expected_baseline: Expected/calibrated baseline value
            timestamps: Optional timestamps for readings
            tag_id: Sensor tag identifier

        Returns:
            DriftResult with detection details
        """
        start_time = datetime.now()

        if len(sensor_history) < self.config.slow_drift_window:
            return DriftResult(
                detected=False,
                description="Insufficient data for slow drift detection",
                affected_tag=tag_id,
            )

        values = np.array(sensor_history[-self.config.slow_drift_window:])
        x = np.arange(len(values))

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        r_squared = r_value ** 2

        # Calculate drift metrics
        current_mean = np.mean(values[-10:])
        drift_from_baseline = current_mean - expected_baseline
        drift_pct = abs(drift_from_baseline / expected_baseline * 100) if expected_baseline != 0 else 0

        # Determine if significant drift
        threshold_exceeded = drift_pct > self.config.slow_drift_threshold_pct
        trend_significant = r_squared >= self.config.trend_min_r_squared

        detected = threshold_exceeded and trend_significant
        drift_type = None

        if detected:
            if slope > 0:
                drift_type = DriftType.INCREASING_TREND
            else:
                drift_type = DriftType.DECREASING_TREND

            if abs(drift_pct) < self.config.slow_drift_threshold_pct * 2:
                drift_type = DriftType.SLOW_BIAS

        # Calculate confidence
        confidence = min(1.0, r_squared * (drift_pct / self.config.slow_drift_threshold_pct))
        confidence = max(0.0, min(1.0, confidence))

        result = DriftResult(
            detected=detected,
            drift_type=drift_type,
            magnitude=float(drift_from_baseline),
            time_detected=datetime.now(timezone.utc),
            drift_start_time=timestamps[0] if timestamps else None,
            confidence=confidence,
            affected_tag=tag_id,
            description=f"{'Detected' if detected else 'No'} slow drift: {drift_pct:.2f}% from baseline",
            metrics={
                "slope": float(slope),
                "r_squared": float(r_squared),
                "baseline": float(expected_baseline),
                "current_mean": float(current_mean),
                "drift_pct": float(drift_pct),
                "p_value": float(p_value),
                "std_error": float(std_err),
            }
        )

        result.provenance_hash = self._calculate_provenance(result, sensor_history)
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Slow drift detection completed in {duration_ms:.2f}ms")

        return result

    def detect_step_change(
        self,
        sensor_values: List[float],
        threshold: Optional[float] = None,
        timestamps: Optional[List[datetime]] = None,
        tag_id: str = ""
    ) -> DriftResult:
        """
        Detect step changes in sensor readings.

        Uses z-score analysis with sliding window to detect sudden shifts.
        Zero-hallucination: Uses deterministic statistical methods.

        Args:
            sensor_values: Sensor values to analyze
            threshold: Optional override for sigma threshold
            timestamps: Optional timestamps for readings
            tag_id: Sensor tag identifier

        Returns:
            DriftResult with detection details
        """
        start_time = datetime.now()
        sigma_threshold = threshold or self.config.step_change_sigma

        if len(sensor_values) < self.config.step_change_window * 2:
            return DriftResult(
                detected=False,
                description="Insufficient data for step change detection",
                affected_tag=tag_id,
            )

        values = np.array(sensor_values)
        window = self.config.step_change_window

        # Calculate rolling statistics
        rolling_mean = np.convolve(values, np.ones(window)/window, mode='valid')
        rolling_std = np.array([
            np.std(values[i:i+window]) for i in range(len(values) - window + 1)
        ])

        # Prevent division by zero
        rolling_std = np.maximum(rolling_std, 1e-10)

        # Calculate differences between adjacent windows
        if len(rolling_mean) >= 2:
            mean_diffs = np.diff(rolling_mean)
            window_stds = rolling_std[:-1]

            # Z-score of changes
            z_scores = mean_diffs / window_stds

            # Find step changes
            step_indices = np.where(np.abs(z_scores) > sigma_threshold)[0]

            if len(step_indices) > 0:
                max_z_idx = step_indices[np.argmax(np.abs(z_scores[step_indices]))]
                max_z = float(np.abs(z_scores[max_z_idx]))
                step_magnitude = float(mean_diffs[max_z_idx])

                confidence = min(1.0, max_z / (sigma_threshold * 2))

                result = DriftResult(
                    detected=True,
                    drift_type=DriftType.STEP_CHANGE,
                    magnitude=step_magnitude,
                    time_detected=datetime.now(timezone.utc),
                    drift_start_time=timestamps[max_z_idx] if timestamps else None,
                    confidence=confidence,
                    affected_tag=tag_id,
                    description=f"Step change detected: magnitude={step_magnitude:.4f}, z-score={max_z:.2f}",
                    metrics={
                        "max_z_score": max_z,
                        "step_index": int(max_z_idx),
                        "pre_step_mean": float(rolling_mean[max_z_idx]),
                        "post_step_mean": float(rolling_mean[max_z_idx + 1]),
                        "threshold_sigma": float(sigma_threshold),
                    }
                )
                result.provenance_hash = self._calculate_provenance(result, sensor_values)
                return result

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.debug(f"Step change detection completed in {duration_ms:.2f}ms")

        return DriftResult(
            detected=False,
            description="No step change detected",
            affected_tag=tag_id,
        )

    def detect_flatline(
        self,
        sensor_values: List[float],
        tolerance: Optional[float] = None,
        duration: Optional[int] = None,
        timestamps: Optional[List[datetime]] = None,
        tag_id: str = ""
    ) -> DriftResult:
        """
        Detect flatline conditions (stuck sensor).

        Uses variance analysis to detect constant readings.
        Zero-hallucination: Uses deterministic variance calculation.

        Args:
            sensor_values: Sensor values to analyze
            tolerance: Optional override for variance tolerance
            duration: Optional override for minimum flatline duration
            timestamps: Optional timestamps for readings
            tag_id: Sensor tag identifier

        Returns:
            DriftResult with detection details
        """
        start_time = datetime.now()
        tol = tolerance or self.config.flatline_tolerance
        min_dur = duration or self.config.flatline_min_duration

        if len(sensor_values) < min_dur:
            return DriftResult(
                detected=False,
                description="Insufficient data for flatline detection",
                affected_tag=tag_id,
            )

        values = np.array(sensor_values)

        # Check rolling variance
        flatline_start = None
        flatline_length = 0
        flatline_value = None

        for i in range(len(values) - min_dur + 1):
            window = values[i:i + min_dur]
            variance = np.var(window)

            if variance < tol:
                if flatline_start is None:
                    flatline_start = i
                    flatline_value = float(np.mean(window))
                flatline_length = i - flatline_start + min_dur
            else:
                if flatline_length >= min_dur:
                    break
                flatline_start = None
                flatline_length = 0

        if flatline_length >= min_dur and flatline_start is not None:
            confidence = 1.0 - (variance / tol) if variance < tol else 0.5

            result = DriftResult(
                detected=True,
                drift_type=DriftType.FLATLINE,
                magnitude=0.0,
                time_detected=datetime.now(timezone.utc),
                drift_start_time=timestamps[flatline_start] if timestamps else None,
                confidence=confidence,
                affected_tag=tag_id,
                description=f"Flatline detected: value={flatline_value:.4f}, duration={flatline_length} samples",
                metrics={
                    "flatline_value": flatline_value,
                    "flatline_duration": flatline_length,
                    "variance": float(variance) if 'variance' in locals() else 0.0,
                    "tolerance": float(tol),
                    "start_index": int(flatline_start),
                }
            )
            result.provenance_hash = self._calculate_provenance(result, sensor_values)
            return result

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.debug(f"Flatline detection completed in {duration_ms:.2f}ms")

        return DriftResult(
            detected=False,
            description="No flatline detected",
            affected_tag=tag_id,
        )

    def monitor_reagent_flow(
        self,
        flow_values: List[float],
        timestamps: Optional[List[datetime]] = None,
        tag_id: str = ""
    ) -> DriftResult:
        """
        Monitor reagent flow for analyzer health.

        Detects low flow, high flow, and erratic flow conditions.
        Zero-hallucination: Uses deterministic threshold and CV analysis.

        Args:
            flow_values: Reagent flow rate values (mL/min)
            timestamps: Optional timestamps for readings
            tag_id: Reagent flow tag identifier

        Returns:
            DriftResult with flow health status
        """
        start_time = datetime.now()

        if len(flow_values) < 5:
            return DriftResult(
                detected=False,
                description="Insufficient data for reagent flow monitoring",
                affected_tag=tag_id,
            )

        values = np.array(flow_values)
        mean_flow = np.mean(values)
        std_flow = np.std(values)
        cv = std_flow / mean_flow if mean_flow > 0 else 0

        # Check for various fault conditions
        low_flow = np.any(values < self.config.reagent_flow_min)
        high_flow = np.any(values > self.config.reagent_flow_max)
        erratic = cv > self.config.reagent_flow_cv_threshold

        drift_type = None
        detected = False
        description = "Reagent flow within normal range"

        if low_flow:
            detected = True
            drift_type = DriftType.REAGENT_FLOW_LOW
            min_val = float(np.min(values))
            description = f"Low reagent flow detected: min={min_val:.2f} mL/min"
        elif high_flow:
            detected = True
            drift_type = DriftType.REAGENT_FLOW_HIGH
            max_val = float(np.max(values))
            description = f"High reagent flow detected: max={max_val:.2f} mL/min"
        elif erratic:
            detected = True
            drift_type = DriftType.REAGENT_FLOW_ERRATIC
            description = f"Erratic reagent flow detected: CV={cv:.2%}"

        # Calculate confidence
        if detected:
            if drift_type == DriftType.REAGENT_FLOW_LOW:
                min_val = np.min(values)
                confidence = 1.0 - (min_val / self.config.reagent_flow_min)
            elif drift_type == DriftType.REAGENT_FLOW_HIGH:
                max_val = np.max(values)
                confidence = (max_val - self.config.reagent_flow_max) / self.config.reagent_flow_max
            else:
                confidence = cv / self.config.reagent_flow_cv_threshold

            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 0.0

        result = DriftResult(
            detected=detected,
            drift_type=drift_type,
            magnitude=float(mean_flow),
            time_detected=datetime.now(timezone.utc),
            confidence=confidence,
            affected_tag=tag_id,
            description=description,
            metrics={
                "mean_flow": float(mean_flow),
                "std_flow": float(std_flow),
                "cv": float(cv),
                "min_flow": float(np.min(values)),
                "max_flow": float(np.max(values)),
                "threshold_min": self.config.reagent_flow_min,
                "threshold_max": self.config.reagent_flow_max,
            }
        )

        result.provenance_hash = self._calculate_provenance(result, flow_values)
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Reagent flow monitoring completed in {duration_ms:.2f}ms")

        return result

    def detect_oscillation(
        self,
        sensor_values: List[float],
        timestamps: Optional[List[datetime]] = None,
        tag_id: str = ""
    ) -> DriftResult:
        """
        Detect oscillating sensor behavior.

        Uses peak detection to identify regular oscillations.
        Zero-hallucination: Uses deterministic peak finding.

        Args:
            sensor_values: Sensor values to analyze
            timestamps: Optional timestamps for readings
            tag_id: Sensor tag identifier

        Returns:
            DriftResult with oscillation detection
        """
        if len(sensor_values) < 20:
            return DriftResult(
                detected=False,
                description="Insufficient data for oscillation detection",
                affected_tag=tag_id,
            )

        values = np.array(sensor_values)

        # Normalize values for peak detection
        normalized = (values - np.mean(values)) / (np.std(values) + 1e-10)

        # Find peaks and troughs
        peaks, peak_props = find_peaks(normalized, height=0.5, distance=3)
        troughs, trough_props = find_peaks(-normalized, height=0.5, distance=3)

        # Check for regular oscillation pattern
        if len(peaks) >= 3 and len(troughs) >= 3:
            peak_intervals = np.diff(peaks)
            trough_intervals = np.diff(troughs)

            peak_cv = np.std(peak_intervals) / np.mean(peak_intervals) if np.mean(peak_intervals) > 0 else 1.0
            trough_cv = np.std(trough_intervals) / np.mean(trough_intervals) if np.mean(trough_intervals) > 0 else 1.0

            # Regular oscillation if intervals are consistent
            is_regular = peak_cv < 0.3 and trough_cv < 0.3
            has_enough_cycles = len(peaks) >= 3

            if is_regular and has_enough_cycles:
                avg_period = np.mean(peak_intervals)
                amplitude = (np.mean(values[peaks]) - np.mean(values[troughs])) / 2
                confidence = min(1.0, (1 - (peak_cv + trough_cv) / 2))

                result = DriftResult(
                    detected=True,
                    drift_type=DriftType.OSCILLATING,
                    magnitude=float(amplitude),
                    time_detected=datetime.now(timezone.utc),
                    confidence=confidence,
                    affected_tag=tag_id,
                    description=f"Oscillation detected: period={avg_period:.1f} samples, amplitude={amplitude:.4f}",
                    metrics={
                        "peak_count": len(peaks),
                        "trough_count": len(troughs),
                        "avg_period": float(avg_period),
                        "amplitude": float(amplitude),
                        "peak_cv": float(peak_cv),
                        "trough_cv": float(trough_cv),
                    }
                )
                result.provenance_hash = self._calculate_provenance(result, sensor_values)
                return result

        return DriftResult(
            detected=False,
            description="No significant oscillation detected",
            affected_tag=tag_id,
        )

    def _calculate_provenance(
        self,
        result: DriftResult,
        values: List[float]
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{result.result_id}"
            f"{result.drift_type}"
            f"{result.time_detected.isoformat()}"
            f"{len(values)}"
            f"{sum(values):.10f}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
