"""
Calibration Tracker - Prediction interval coverage and drift detection

This module implements calibration tracking for uncertainty quantification,
monitoring prediction interval coverage and detecting model drift.
All calculations are deterministic with SHA-256 provenance tracking.

Key Components:
    - CalibrationTracker: Main tracker for interval coverage
    - DriftDetector: Detect distribution drift and trigger retraining
    - ReliabilityDiagramGenerator: Generate reliability diagrams

Reference Standards:
    - ISO 5725 (Accuracy and precision)
    - NIST SP 800-161 (Supply Chain Risk Management)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple, Deque
from uuid import UUID, uuid4
from enum import Enum
import math

from .uq_schemas import (
    ProvenanceRecord,
    PredictionInterval,
    CalibrationMetrics,
    CalibrationStatus,
    ReliabilityDiagram,
    ReliabilityDiagramPoint,
)


class DriftType(str, Enum):
    """Types of drift that can be detected."""
    NONE = "none"
    MEAN_SHIFT = "mean_shift"
    VARIANCE_CHANGE = "variance_change"
    COVERAGE_DRIFT = "coverage_drift"
    SUDDEN_DRIFT = "sudden_drift"
    GRADUAL_DRIFT = "gradual_drift"


class DriftAlert:
    """Alert for detected drift."""

    def __init__(
        self,
        drift_type: DriftType,
        severity: str,
        metric_name: str,
        current_value: Decimal,
        expected_value: Decimal,
        timestamp: datetime,
        description: str
    ):
        self.alert_id = uuid4()
        self.drift_type = drift_type
        self.severity = severity
        self.metric_name = metric_name
        self.current_value = current_value
        self.expected_value = expected_value
        self.timestamp = timestamp
        self.description = description
        self.acknowledged = False


class CalibrationTracker:
    """
    Track prediction interval calibration over time - ZERO HALLUCINATION.

    Monitors:
    - PICP (Prediction Interval Coverage Probability)
    - Mean/Normalized Prediction Interval Width
    - Calibration error and drift
    - Coverage by confidence level

    All calculations are deterministic with complete provenance tracking.
    """

    def __init__(
        self,
        model_name: str,
        variable_name: str,
        window_size: int = 100,
        min_samples: int = 20
    ):
        """
        Initialize calibration tracker.

        Args:
            model_name: Name of the forecasting model
            variable_name: Variable being tracked
            window_size: Rolling window size for metrics
            min_samples: Minimum samples before computing metrics
        """
        self.model_name = model_name
        self.variable_name = variable_name
        self.window_size = window_size
        self.min_samples = min_samples

        # Rolling windows for different confidence levels
        self._coverage_records: Dict[str, Deque[bool]] = {}
        self._width_records: Deque[Decimal] = deque(maxlen=window_size)
        self._normalized_width_records: Deque[Decimal] = deque(maxlen=window_size)
        self._error_records: Deque[Decimal] = deque(maxlen=window_size)

        # Historical metrics
        self._metrics_history: List[CalibrationMetrics] = []

        # Tracking timestamps
        self._first_record_time: Optional[datetime] = None
        self._last_record_time: Optional[datetime] = None
        self._total_predictions: int = 0

    def record_observation(
        self,
        prediction: PredictionInterval,
        actual_value: Decimal
    ) -> None:
        """
        Record an observation for calibration tracking - DETERMINISTIC.

        Args:
            prediction: Prediction interval that was made
            actual_value: Actual observed value
        """
        # Initialize confidence level tracking if needed
        conf_key = str(prediction.confidence_level)
        if conf_key not in self._coverage_records:
            self._coverage_records[conf_key] = deque(maxlen=self.window_size)

        # Check if actual value is in interval
        covered = prediction.contains(actual_value)
        self._coverage_records[conf_key].append(covered)

        # Record interval width
        self._width_records.append(prediction.interval_width)

        # Record normalized width
        if prediction.point_estimate != 0:
            normalized = prediction.interval_width / abs(prediction.point_estimate)
        else:
            normalized = prediction.interval_width
        self._normalized_width_records.append(normalized)

        # Record prediction error
        error = abs(actual_value - prediction.point_estimate)
        self._error_records.append(error)

        # Update timestamps
        if self._first_record_time is None:
            self._first_record_time = prediction.timestamp
        self._last_record_time = prediction.timestamp
        self._total_predictions += 1

    def compute_metrics(
        self,
        target_coverage: Decimal = Decimal("0.90")
    ) -> Optional[CalibrationMetrics]:
        """
        Compute calibration metrics - DETERMINISTIC.

        Args:
            target_coverage: Target coverage level for calibration

        Returns:
            CalibrationMetrics or None if insufficient samples
        """
        start_time = time.time()

        conf_key = str(target_coverage)
        if conf_key not in self._coverage_records:
            return None

        coverage_list = list(self._coverage_records[conf_key])
        if len(coverage_list) < self.min_samples:
            return None

        # Compute PICP (Prediction Interval Coverage Probability)
        covered_count = sum(1 for c in coverage_list if c)
        picp = Decimal(str(covered_count)) / Decimal(str(len(coverage_list)))

        # Compute MPIW (Mean Prediction Interval Width)
        width_list = list(self._width_records)
        if width_list:
            mpiw = sum(width_list) / Decimal(str(len(width_list)))
        else:
            mpiw = Decimal("0")

        # Compute NMPIW (Normalized MPIW)
        norm_width_list = list(self._normalized_width_records)
        if norm_width_list:
            nmpiw = sum(norm_width_list) / Decimal(str(len(norm_width_list)))
        else:
            nmpiw = Decimal("0")

        # Compute Winkler score (penalized interval width)
        winkler = self._compute_winkler_score(target_coverage)

        # Compute CRPS approximation
        crps = self._compute_crps_approximation()

        # Calibration error
        calibration_error = abs(picp - target_coverage)

        # Determine status
        tolerance = Decimal("0.05")
        if calibration_error <= tolerance:
            status = CalibrationStatus.WELL_CALIBRATED
        elif picp > target_coverage + tolerance:
            status = CalibrationStatus.UNDER_CONFIDENT
        elif picp < target_coverage - tolerance:
            status = CalibrationStatus.OVER_CONFIDENT
        else:
            status = CalibrationStatus.WELL_CALIBRATED

        # Check for drift
        drift_detected = self._check_drift()
        if drift_detected:
            status = CalibrationStatus.DRIFT_DETECTED

        # Check if retraining is needed
        retraining_recommended = (
            calibration_error > Decimal("0.10") or
            drift_detected
        )
        if retraining_recommended:
            status = CalibrationStatus.REQUIRES_RETRAINING

        # Create provenance
        computation_time_ms = (time.time() - start_time) * 1000
        provenance = ProvenanceRecord.create(
            calculation_type="calibration_metrics_computation",
            inputs={
                "model_name": self.model_name,
                "variable_name": self.variable_name,
                "target_coverage": str(target_coverage),
                "num_samples": len(coverage_list),
                "window_size": self.window_size
            },
            outputs={
                "picp": str(picp),
                "mpiw": str(mpiw),
                "nmpiw": str(nmpiw),
                "calibration_error": str(calibration_error),
                "status": status.value
            },
            computation_time_ms=computation_time_ms
        )

        metrics = CalibrationMetrics(
            model_name=self.model_name,
            variable_name=self.variable_name,
            evaluation_period_start=self._first_record_time or datetime.utcnow(),
            evaluation_period_end=self._last_record_time or datetime.utcnow(),
            num_predictions=len(coverage_list),
            picp=picp,
            target_coverage=target_coverage,
            mpiw=mpiw,
            nmpiw=nmpiw,
            winkler_score=winkler,
            crps=crps,
            calibration_error=calibration_error,
            status=status,
            drift_detected=drift_detected,
            retraining_recommended=retraining_recommended,
            provenance=provenance
        )

        self._metrics_history.append(metrics)
        return metrics

    def _compute_winkler_score(
        self,
        target_coverage: Decimal
    ) -> Decimal:
        """
        Compute Winkler score - DETERMINISTIC.

        Winkler score penalizes both wide intervals and non-coverage.
        """
        # Simplified Winkler score based on stored data
        alpha = Decimal("1") - target_coverage
        width_list = list(self._width_records)

        if not width_list:
            return Decimal("0")

        # Average width as base score
        avg_width = sum(width_list) / Decimal(str(len(width_list)))

        # Penalty for non-coverage (simplified)
        conf_key = str(target_coverage)
        if conf_key in self._coverage_records:
            coverage_list = list(self._coverage_records[conf_key])
            non_covered = sum(1 for c in coverage_list if not c)
            penalty = Decimal(str(non_covered)) * avg_width * (Decimal("2") / alpha)
        else:
            penalty = Decimal("0")

        total = avg_width + penalty / Decimal(str(len(width_list)))
        return total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _compute_crps_approximation(self) -> Decimal:
        """
        Compute CRPS approximation - DETERMINISTIC.

        CRPS = integral of (F(y) - 1{y >= actual})^2 dy
        Approximated using prediction error and interval width.
        """
        error_list = list(self._error_records)
        width_list = list(self._width_records)

        if not error_list or not width_list:
            return Decimal("0")

        # Simple approximation: mean error + penalty for wide intervals
        mean_error = sum(error_list) / Decimal(str(len(error_list)))
        mean_width = sum(width_list) / Decimal(str(len(width_list)))

        # CRPS approximation
        crps = mean_error + Decimal("0.1") * mean_width
        return crps.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _check_drift(self) -> bool:
        """
        Check for drift using simple statistical tests - DETERMINISTIC.

        Returns True if drift is detected.
        """
        # Need at least 2 * min_samples for comparison
        error_list = list(self._error_records)
        if len(error_list) < 2 * self.min_samples:
            return False

        # Split into two halves
        mid = len(error_list) // 2
        first_half = error_list[:mid]
        second_half = error_list[mid:]

        # Compare means
        mean_first = sum(first_half) / Decimal(str(len(first_half)))
        mean_second = sum(second_half) / Decimal(str(len(second_half)))

        # Compute pooled std
        var_first = sum((x - mean_first) ** 2 for x in first_half) / Decimal(str(len(first_half)))
        var_second = sum((x - mean_second) ** 2 for x in second_half) / Decimal(str(len(second_half)))

        pooled_std = ((var_first + var_second) / Decimal("2")).sqrt() if var_first + var_second > 0 else Decimal("0.001")

        # Simple drift detection: significant change in mean error
        mean_change = abs(mean_second - mean_first)
        threshold = Decimal("2") * pooled_std  # 2 standard deviations

        return mean_change > threshold

    def get_coverage_by_confidence(self) -> Dict[str, Decimal]:
        """Get PICP for each tracked confidence level - DETERMINISTIC."""
        result = {}
        for conf_key, records in self._coverage_records.items():
            records_list = list(records)
            if records_list:
                covered = sum(1 for c in records_list if c)
                picp = Decimal(str(covered)) / Decimal(str(len(records_list)))
                result[conf_key] = picp.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        return result

    def get_metrics_history(self) -> List[CalibrationMetrics]:
        """Get historical metrics."""
        return self._metrics_history.copy()

    def reset(self) -> None:
        """Reset all tracking data."""
        self._coverage_records.clear()
        self._width_records.clear()
        self._normalized_width_records.clear()
        self._error_records.clear()
        self._metrics_history.clear()
        self._first_record_time = None
        self._last_record_time = None
        self._total_predictions = 0


class DriftDetector:
    """
    Drift detection for uncertainty models - ZERO HALLUCINATION.

    Implements multiple drift detection methods:
    - Page-Hinkley test
    - ADWIN (ADaptive WINdowing)
    - CUSUM (Cumulative Sum)

    All calculations are deterministic for reproducibility.
    """

    def __init__(
        self,
        method: str = "page_hinkley",
        threshold: Decimal = Decimal("0.05"),
        window_size: int = 50
    ):
        """
        Initialize drift detector.

        Args:
            method: Detection method (page_hinkley, cusum, adwin)
            threshold: Detection threshold
            window_size: Window size for methods that use it
        """
        self.method = method
        self.threshold = threshold
        self.window_size = window_size

        # Page-Hinkley state
        self._ph_sum = Decimal("0")
        self._ph_min = Decimal("0")
        self._ph_mean = Decimal("0")
        self._ph_count = 0

        # CUSUM state
        self._cusum_pos = Decimal("0")
        self._cusum_neg = Decimal("0")
        self._cusum_mean = Decimal("0")
        self._cusum_std = Decimal("1")
        self._cusum_count = 0

        # ADWIN state
        self._adwin_window: Deque[Decimal] = deque(maxlen=window_size * 2)

        # Alert history
        self._alerts: List[DriftAlert] = []

    def update(
        self,
        value: Decimal,
        timestamp: Optional[datetime] = None
    ) -> Optional[DriftAlert]:
        """
        Update detector with new value - DETERMINISTIC.

        Args:
            value: New observation value
            timestamp: Observation timestamp

        Returns:
            DriftAlert if drift detected, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        if self.method == "page_hinkley":
            return self._update_page_hinkley(value, timestamp)
        elif self.method == "cusum":
            return self._update_cusum(value, timestamp)
        elif self.method == "adwin":
            return self._update_adwin(value, timestamp)
        else:
            return self._update_page_hinkley(value, timestamp)

    def _update_page_hinkley(
        self,
        value: Decimal,
        timestamp: datetime
    ) -> Optional[DriftAlert]:
        """
        Page-Hinkley test for drift detection - DETERMINISTIC.

        Detects mean shift by tracking cumulative deviation.
        """
        self._ph_count += 1

        # Update running mean
        old_mean = self._ph_mean
        self._ph_mean = old_mean + (value - old_mean) / Decimal(str(self._ph_count))

        # Update cumulative sum
        self._ph_sum = self._ph_sum + (value - self._ph_mean - self.threshold)

        # Update minimum
        if self._ph_sum < self._ph_min:
            self._ph_min = self._ph_sum

        # Compute test statistic
        ph_stat = self._ph_sum - self._ph_min

        # Detect drift
        drift_threshold = Decimal("10") * self.threshold * Decimal(str(self._ph_count)).sqrt()

        if ph_stat > drift_threshold and self._ph_count > self.window_size:
            alert = DriftAlert(
                drift_type=DriftType.MEAN_SHIFT,
                severity="high" if ph_stat > drift_threshold * Decimal("2") else "medium",
                metric_name="page_hinkley_statistic",
                current_value=ph_stat,
                expected_value=drift_threshold,
                timestamp=timestamp,
                description=f"Mean shift detected: PH stat {ph_stat:.3f} exceeds threshold {drift_threshold:.3f}"
            )
            self._alerts.append(alert)

            # Reset detector after alert
            self._reset_page_hinkley()
            return alert

        return None

    def _reset_page_hinkley(self) -> None:
        """Reset Page-Hinkley state."""
        self._ph_sum = Decimal("0")
        self._ph_min = Decimal("0")

    def _update_cusum(
        self,
        value: Decimal,
        timestamp: datetime
    ) -> Optional[DriftAlert]:
        """
        CUSUM test for drift detection - DETERMINISTIC.

        Detects shifts in both directions.
        """
        self._cusum_count += 1

        # Update running statistics
        if self._cusum_count <= self.window_size:
            old_mean = self._cusum_mean
            self._cusum_mean = old_mean + (value - old_mean) / Decimal(str(self._cusum_count))

            # Update variance
            if self._cusum_count > 1:
                old_var = self._cusum_std ** 2
                new_var = old_var + ((value - old_mean) * (value - self._cusum_mean) - old_var) / Decimal(str(self._cusum_count))
                self._cusum_std = new_var.sqrt() if new_var > 0 else Decimal("0.001")

            return None

        # Standardize value
        if self._cusum_std > 0:
            z = (value - self._cusum_mean) / self._cusum_std
        else:
            z = value - self._cusum_mean

        # Update CUSUM statistics
        k = Decimal("0.5")  # Allowance parameter
        self._cusum_pos = max(Decimal("0"), self._cusum_pos + z - k)
        self._cusum_neg = max(Decimal("0"), self._cusum_neg - z - k)

        # Detection threshold
        h = Decimal("4")  # Standard CUSUM threshold

        if self._cusum_pos > h:
            alert = DriftAlert(
                drift_type=DriftType.MEAN_SHIFT,
                severity="high",
                metric_name="cusum_positive",
                current_value=self._cusum_pos,
                expected_value=h,
                timestamp=timestamp,
                description=f"Upward shift detected: CUSUM+ {self._cusum_pos:.3f} > {h}"
            )
            self._alerts.append(alert)
            self._cusum_pos = Decimal("0")
            return alert

        if self._cusum_neg > h:
            alert = DriftAlert(
                drift_type=DriftType.MEAN_SHIFT,
                severity="high",
                metric_name="cusum_negative",
                current_value=self._cusum_neg,
                expected_value=h,
                timestamp=timestamp,
                description=f"Downward shift detected: CUSUM- {self._cusum_neg:.3f} > {h}"
            )
            self._alerts.append(alert)
            self._cusum_neg = Decimal("0")
            return alert

        return None

    def _update_adwin(
        self,
        value: Decimal,
        timestamp: datetime
    ) -> Optional[DriftAlert]:
        """
        ADWIN test for drift detection - DETERMINISTIC.

        Adaptive windowing algorithm for concept drift.
        """
        self._adwin_window.append(value)

        if len(self._adwin_window) < self.window_size:
            return None

        # Try different split points
        window_list = list(self._adwin_window)
        n = len(window_list)

        for split in range(self.window_size // 2, n - self.window_size // 2):
            left = window_list[:split]
            right = window_list[split:]

            # Compute means
            mean_left = sum(left) / Decimal(str(len(left)))
            mean_right = sum(right) / Decimal(str(len(right)))

            # Compute variances
            var_left = sum((x - mean_left) ** 2 for x in left) / Decimal(str(len(left)))
            var_right = sum((x - mean_right) ** 2 for x in right) / Decimal(str(len(right)))

            # Hoeffding bound
            n_left = Decimal(str(len(left)))
            n_right = Decimal(str(len(right)))
            m = Decimal("1") / (Decimal("1") / n_left + Decimal("1") / n_right)

            delta = self.threshold
            epsilon = (
                (Decimal("1") / (Decimal("2") * m)) *
                Decimal(str(math.log(4 / float(delta))))
            ).sqrt()

            if abs(mean_left - mean_right) > epsilon:
                alert = DriftAlert(
                    drift_type=DriftType.GRADUAL_DRIFT,
                    severity="medium",
                    metric_name="adwin_mean_diff",
                    current_value=abs(mean_left - mean_right),
                    expected_value=epsilon,
                    timestamp=timestamp,
                    description=f"ADWIN drift: mean diff {abs(mean_left - mean_right):.3f} > bound {epsilon:.3f}"
                )
                self._alerts.append(alert)

                # Shrink window
                while len(self._adwin_window) > self.window_size:
                    self._adwin_window.popleft()

                return alert

        return None

    def get_alerts(
        self,
        since: Optional[datetime] = None,
        severity: Optional[str] = None
    ) -> List[DriftAlert]:
        """Get drift alerts filtered by time and severity."""
        alerts = self._alerts

        if since is not None:
            alerts = [a for a in alerts if a.timestamp >= since]

        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def reset(self) -> None:
        """Reset all detector state."""
        self._ph_sum = Decimal("0")
        self._ph_min = Decimal("0")
        self._ph_mean = Decimal("0")
        self._ph_count = 0

        self._cusum_pos = Decimal("0")
        self._cusum_neg = Decimal("0")
        self._cusum_mean = Decimal("0")
        self._cusum_std = Decimal("1")
        self._cusum_count = 0

        self._adwin_window.clear()
        self._alerts.clear()


class ReliabilityDiagramGenerator:
    """
    Generate reliability diagrams for calibration assessment - ZERO HALLUCINATION.

    Reliability diagrams show the relationship between predicted
    probabilities and observed frequencies for interval calibration.
    """

    def __init__(self, num_bins: int = 10):
        """
        Initialize reliability diagram generator.

        Args:
            num_bins: Number of bins for the diagram
        """
        self.num_bins = num_bins
        self._predictions: List[Tuple[Decimal, bool]] = []

    def add_prediction(
        self,
        predicted_probability: Decimal,
        actual_outcome: bool
    ) -> None:
        """
        Add a prediction for reliability diagram.

        Args:
            predicted_probability: Predicted probability (confidence level)
            actual_outcome: Whether prediction was correct (in interval)
        """
        self._predictions.append((predicted_probability, actual_outcome))

    def generate(
        self,
        model_name: str = "unknown",
        variable_name: str = "unknown"
    ) -> Optional[ReliabilityDiagram]:
        """
        Generate reliability diagram - DETERMINISTIC.

        Returns:
            ReliabilityDiagram or None if insufficient data
        """
        start_time = time.time()

        if len(self._predictions) < self.num_bins:
            return None

        # Sort predictions by probability
        sorted_preds = sorted(self._predictions, key=lambda x: x[0])

        # Create bins
        bin_boundaries = [
            Decimal(str(i)) / Decimal(str(self.num_bins))
            for i in range(self.num_bins + 1)
        ]

        points = []
        for i in range(self.num_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]

            # Get predictions in this bin
            bin_preds = [
                (p, o) for p, o in sorted_preds
                if lower <= p < upper or (i == self.num_bins - 1 and p == upper)
            ]

            if bin_preds:
                mean_prob = sum(p for p, _ in bin_preds) / Decimal(str(len(bin_preds)))
                observed_freq = Decimal(str(sum(1 for _, o in bin_preds if o))) / Decimal(str(len(bin_preds)))

                points.append(ReliabilityDiagramPoint(
                    predicted_probability=mean_prob.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
                    observed_frequency=observed_freq.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
                    num_samples=len(bin_preds)
                ))

        if not points:
            return None

        # Compute ECE (Expected Calibration Error)
        total_samples = len(self._predictions)
        ece = Decimal("0")
        mce = Decimal("0")

        for point in points:
            bin_weight = Decimal(str(point.num_samples)) / Decimal(str(total_samples))
            bin_error = abs(point.predicted_probability - point.observed_frequency)
            ece += bin_weight * bin_error
            if bin_error > mce:
                mce = bin_error

        computation_time_ms = (time.time() - start_time) * 1000
        provenance = ProvenanceRecord.create(
            calculation_type="reliability_diagram_generation",
            inputs={
                "model_name": model_name,
                "variable_name": variable_name,
                "num_bins": self.num_bins,
                "total_predictions": len(self._predictions)
            },
            outputs={
                "num_points": len(points),
                "ece": str(ece),
                "mce": str(mce)
            },
            computation_time_ms=computation_time_ms
        )

        return ReliabilityDiagram(
            model_name=model_name,
            variable_name=variable_name,
            points=points,
            num_bins=self.num_bins,
            total_samples=total_samples,
            expected_calibration_error=ece.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
            maximum_calibration_error=mce.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
            provenance=provenance
        )

    def reset(self) -> None:
        """Reset all prediction data."""
        self._predictions.clear()


class CalibrationReportGenerator:
    """
    Generate comprehensive calibration reports - ZERO HALLUCINATION.

    Combines metrics, reliability diagrams, and drift detection
    into a complete calibration assessment.
    """

    def __init__(
        self,
        tracker: CalibrationTracker,
        drift_detector: DriftDetector,
        diagram_generator: ReliabilityDiagramGenerator
    ):
        """Initialize report generator."""
        self.tracker = tracker
        self.drift_detector = drift_detector
        self.diagram_generator = diagram_generator

    def generate_report(
        self,
        confidence_levels: List[Decimal] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive calibration report - DETERMINISTIC.

        Returns:
            Dictionary containing complete calibration assessment
        """
        if confidence_levels is None:
            confidence_levels = [Decimal("0.50"), Decimal("0.80"), Decimal("0.90"), Decimal("0.95")]

        start_time = time.time()

        # Compute metrics for each confidence level
        metrics_by_level = {}
        for level in confidence_levels:
            metrics = self.tracker.compute_metrics(target_coverage=level)
            if metrics:
                metrics_by_level[str(level)] = {
                    "picp": str(metrics.picp),
                    "mpiw": str(metrics.mpiw),
                    "nmpiw": str(metrics.nmpiw),
                    "calibration_error": str(metrics.calibration_error),
                    "status": metrics.status.value
                }

        # Get coverage summary
        coverage_summary = self.tracker.get_coverage_by_confidence()

        # Get drift alerts
        recent_alerts = self.drift_detector.get_alerts(
            since=datetime.utcnow() - timedelta(hours=24)
        )

        # Generate reliability diagram
        diagram = self.diagram_generator.generate(
            model_name=self.tracker.model_name,
            variable_name=self.tracker.variable_name
        )

        # Overall assessment
        overall_status = self._determine_overall_status(metrics_by_level, recent_alerts)

        # Recommendations
        recommendations = self._generate_recommendations(
            metrics_by_level, recent_alerts, overall_status
        )

        computation_time_ms = (time.time() - start_time) * 1000

        report = {
            "report_id": str(uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": self.tracker.model_name,
            "variable_name": self.tracker.variable_name,
            "overall_status": overall_status,
            "metrics_by_confidence_level": metrics_by_level,
            "coverage_summary": {k: str(v) for k, v in coverage_summary.items()},
            "drift_alerts": [
                {
                    "type": a.drift_type.value,
                    "severity": a.severity,
                    "description": a.description,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in recent_alerts
            ],
            "reliability_diagram": {
                "ece": str(diagram.expected_calibration_error) if diagram else None,
                "mce": str(diagram.maximum_calibration_error) if diagram else None,
                "num_points": len(diagram.points) if diagram else 0
            },
            "recommendations": recommendations,
            "computation_time_ms": computation_time_ms,
            "provenance_hash": ProvenanceRecord.compute_hash({
                "metrics": metrics_by_level,
                "coverage": coverage_summary,
                "alerts": len(recent_alerts)
            })
        }

        return report

    def _determine_overall_status(
        self,
        metrics: Dict[str, Any],
        alerts: List[DriftAlert]
    ) -> str:
        """Determine overall calibration status - DETERMINISTIC."""
        if alerts and any(a.severity == "high" for a in alerts):
            return "critical"

        if not metrics:
            return "insufficient_data"

        # Check if any level has poor calibration
        poor_calibration = False
        for level, m in metrics.items():
            error = Decimal(m["calibration_error"])
            if error > Decimal("0.10"):
                poor_calibration = True
                break

        if poor_calibration:
            return "requires_attention"

        if alerts:
            return "monitor"

        return "healthy"

    def _generate_recommendations(
        self,
        metrics: Dict[str, Any],
        alerts: List[DriftAlert],
        status: str
    ) -> List[str]:
        """Generate actionable recommendations - DETERMINISTIC."""
        recommendations = []

        if status == "critical":
            recommendations.append("URGENT: Review recent model predictions for systematic errors")
            recommendations.append("Consider pausing automated decisions until drift is resolved")

        if status == "requires_attention":
            recommendations.append("Model calibration has degraded - schedule retraining")

        for level, m in metrics.items():
            if m["status"] == "over_confident":
                recommendations.append(
                    f"Prediction intervals at {level} confidence are too narrow - "
                    "consider increasing uncertainty estimates"
                )
            elif m["status"] == "under_confident":
                recommendations.append(
                    f"Prediction intervals at {level} confidence are too wide - "
                    "model may be over-estimating uncertainty"
                )

        if alerts:
            recommendations.append(
                f"Investigate {len(alerts)} drift alert(s) from the past 24 hours"
            )

        if not recommendations:
            recommendations.append("Calibration is healthy - continue monitoring")

        return recommendations
