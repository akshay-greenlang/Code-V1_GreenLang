"""
Model Monitoring for GL-003 UNIFIEDSTEAM

Provides comprehensive model monitoring including performance tracking,
drift detection, and alerting for deployed ML models.

Author: GL-003 MLOps Team
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json
import logging
import statistics

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift detected."""
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"
    CONCEPT_DRIFT = "concept_drift"
    LABEL_DRIFT = "label_drift"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertThreshold:
    """
    Threshold configuration for monitoring alerts.

    Defines when alerts should be triggered.
    """
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    direction: str = "above"  # "above" or "below"
    evaluation_window_minutes: int = 60
    min_samples: int = 10
    cooldown_minutes: int = 30

    def check_threshold(self, value: float) -> Optional[AlertSeverity]:
        """Check if value crosses threshold."""
        if self.direction == "above":
            if value >= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value >= self.warning_threshold:
                return AlertSeverity.WARNING
        else:  # below
            if value <= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value <= self.warning_threshold:
                return AlertSeverity.WARNING
        return None


@dataclass
class MonitoringAlert:
    """
    Monitoring alert for model performance issues.
    """
    alert_id: str
    model_id: str
    model_version: str
    alert_type: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    message: str
    metric_name: str
    metric_value: float
    threshold_value: float
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
        }


@dataclass
class PerformanceSnapshot:
    """
    Point-in-time performance snapshot.

    Records model performance metrics at a specific time.
    """
    snapshot_id: str
    model_id: str
    model_version: str
    timestamp: datetime
    sample_count: int

    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None

    # Prediction statistics
    prediction_mean: Optional[float] = None
    prediction_std: Optional[float] = None
    prediction_min: Optional[float] = None
    prediction_max: Optional[float] = None

    # Latency metrics
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None

    # Error rates
    error_rate: Optional[float] = None
    timeout_rate: Optional[float] = None

    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "snapshot_id": self.snapshot_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
            "sample_count": self.sample_count,
        }

        for metric in ["accuracy", "precision", "recall", "f1_score",
                      "false_positive_rate", "false_negative_rate",
                      "prediction_mean", "prediction_std",
                      "prediction_min", "prediction_max",
                      "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
                      "error_rate", "timeout_rate"]:
            value = getattr(self, metric)
            if value is not None:
                result[metric] = value

        if self.custom_metrics:
            result["custom_metrics"] = self.custom_metrics

        return result


@dataclass
class DriftResult:
    """
    Result of drift detection analysis.

    Contains drift scores and statistical tests.
    """
    feature_name: str
    drift_type: DriftType
    timestamp: datetime
    baseline_period: Tuple[datetime, datetime]
    current_period: Tuple[datetime, datetime]

    # Drift scores
    drift_score: float
    drift_detected: bool
    threshold: float

    # Statistical details
    statistical_test: str
    p_value: Optional[float] = None
    test_statistic: Optional[float] = None

    # Distribution statistics
    baseline_mean: Optional[float] = None
    baseline_std: Optional[float] = None
    current_mean: Optional[float] = None
    current_std: Optional[float] = None

    # Samples
    baseline_sample_count: int = 0
    current_sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "drift_type": self.drift_type.value,
            "timestamp": self.timestamp.isoformat(),
            "baseline_period": [
                self.baseline_period[0].isoformat(),
                self.baseline_period[1].isoformat(),
            ],
            "current_period": [
                self.current_period[0].isoformat(),
                self.current_period[1].isoformat(),
            ],
            "drift_score": self.drift_score,
            "drift_detected": self.drift_detected,
            "threshold": self.threshold,
            "statistical_test": self.statistical_test,
            "p_value": self.p_value,
            "test_statistic": self.test_statistic,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "current_mean": self.current_mean,
            "current_std": self.current_std,
            "baseline_sample_count": self.baseline_sample_count,
            "current_sample_count": self.current_sample_count,
        }


@dataclass
class MonitoringReport:
    """
    Comprehensive monitoring report.

    Aggregates performance, drift, and alert information.
    """
    report_id: str
    model_id: str
    model_version: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime

    # Performance
    performance_snapshots: List[PerformanceSnapshot]
    avg_accuracy: Optional[float] = None
    avg_precision: Optional[float] = None
    avg_recall: Optional[float] = None

    # Drift
    drift_results: List[DriftResult] = field(default_factory=list)
    features_with_drift: int = 0

    # Alerts
    alerts_generated: int = 0
    active_alerts: int = 0
    critical_alerts: int = 0

    # Volume
    total_predictions: int = 0
    avg_latency_ms: Optional[float] = None

    # Health score (0-100)
    health_score: float = 100.0
    health_factors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "generated_at": self.generated_at.isoformat(),
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "performance": {
                "snapshots": [s.to_dict() for s in self.performance_snapshots],
                "avg_accuracy": self.avg_accuracy,
                "avg_precision": self.avg_precision,
                "avg_recall": self.avg_recall,
            },
            "drift": {
                "results": [d.to_dict() for d in self.drift_results],
                "features_with_drift": self.features_with_drift,
            },
            "alerts": {
                "generated": self.alerts_generated,
                "active": self.active_alerts,
                "critical": self.critical_alerts,
            },
            "volume": {
                "total_predictions": self.total_predictions,
                "avg_latency_ms": self.avg_latency_ms,
            },
            "health": {
                "score": self.health_score,
                "factors": self.health_factors,
            },
        }


class PerformanceTracker:
    """
    Tracks model performance over time.

    Collects predictions and ground truth to compute metrics.
    """

    def __init__(
        self,
        model_id: str,
        model_version: str,
        snapshot_interval_minutes: int = 60,
    ):
        """
        Initialize performance tracker.

        Args:
            model_id: Model identifier
            model_version: Model version
            snapshot_interval_minutes: Interval for snapshots
        """
        self.model_id = model_id
        self.model_version = model_version
        self.snapshot_interval = timedelta(minutes=snapshot_interval_minutes)

        self._predictions: List[Dict[str, Any]] = []
        self._snapshots: List[PerformanceSnapshot] = []
        self._last_snapshot: Optional[datetime] = None

    def record_prediction(
        self,
        prediction: float,
        ground_truth: Optional[float] = None,
        features: Optional[Dict[str, float]] = None,
        latency_ms: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ):
        """
        Record a prediction for tracking.

        Args:
            prediction: Model prediction
            ground_truth: Actual label (if available)
            features: Input features
            latency_ms: Prediction latency
            timestamp: Prediction timestamp
        """
        record = {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "features": features or {},
            "latency_ms": latency_ms,
            "timestamp": timestamp or datetime.now(timezone.utc),
        }
        self._predictions.append(record)

        # Check if snapshot needed
        if self._should_snapshot():
            self._create_snapshot()

    def _should_snapshot(self) -> bool:
        """Check if a new snapshot should be created."""
        if not self._last_snapshot:
            return len(self._predictions) >= 10
        elapsed = datetime.now(timezone.utc) - self._last_snapshot
        return elapsed >= self.snapshot_interval

    def _create_snapshot(self) -> PerformanceSnapshot:
        """Create performance snapshot from recent predictions."""
        import uuid

        now = datetime.now(timezone.utc)

        # Get predictions since last snapshot
        if self._last_snapshot:
            recent = [
                p for p in self._predictions
                if p["timestamp"] > self._last_snapshot
            ]
        else:
            recent = self._predictions

        if not recent:
            return None

        # Calculate metrics
        predictions = [p["prediction"] for p in recent]
        ground_truths = [
            p["ground_truth"] for p in recent
            if p["ground_truth"] is not None
        ]
        latencies = [
            p["latency_ms"] for p in recent
            if p["latency_ms"] is not None
        ]

        snapshot = PerformanceSnapshot(
            snapshot_id=f"SNAP-{uuid.uuid4().hex[:8].upper()}",
            model_id=self.model_id,
            model_version=self.model_version,
            timestamp=now,
            sample_count=len(recent),
            prediction_mean=statistics.mean(predictions) if predictions else None,
            prediction_std=statistics.stdev(predictions) if len(predictions) > 1 else None,
            prediction_min=min(predictions) if predictions else None,
            prediction_max=max(predictions) if predictions else None,
        )

        # Calculate classification metrics if ground truth available
        if ground_truths and len(ground_truths) == len(predictions):
            threshold = 0.5
            pred_binary = [1 if p >= threshold else 0 for p in predictions]
            actual_binary = [1 if g >= threshold else 0 for g in ground_truths]

            tp = sum(1 for p, a in zip(pred_binary, actual_binary) if p == 1 and a == 1)
            fp = sum(1 for p, a in zip(pred_binary, actual_binary) if p == 1 and a == 0)
            tn = sum(1 for p, a in zip(pred_binary, actual_binary) if p == 0 and a == 0)
            fn = sum(1 for p, a in zip(pred_binary, actual_binary) if p == 0 and a == 1)

            total = tp + fp + tn + fn
            snapshot.accuracy = (tp + tn) / total if total > 0 else None
            snapshot.precision = tp / (tp + fp) if (tp + fp) > 0 else None
            snapshot.recall = tp / (tp + fn) if (tp + fn) > 0 else None

            if snapshot.precision and snapshot.recall:
                snapshot.f1_score = (
                    2 * snapshot.precision * snapshot.recall /
                    (snapshot.precision + snapshot.recall)
                )

            snapshot.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else None
            snapshot.false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else None

        # Calculate latency metrics
        if latencies:
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            snapshot.latency_p50_ms = sorted_latencies[int(n * 0.5)]
            snapshot.latency_p95_ms = sorted_latencies[int(n * 0.95)]
            snapshot.latency_p99_ms = sorted_latencies[int(n * 0.99)]

        self._snapshots.append(snapshot)
        self._last_snapshot = now

        return snapshot

    def get_snapshots(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[PerformanceSnapshot]:
        """Get snapshots within time range."""
        snapshots = self._snapshots

        if start_time:
            snapshots = [s for s in snapshots if s.timestamp >= start_time]
        if end_time:
            snapshots = [s for s in snapshots if s.timestamp <= end_time]

        return snapshots


class DriftDetector:
    """
    Detects feature and prediction drift.

    Compares current distributions against baseline using statistical tests.
    """

    def __init__(
        self,
        baseline_window_days: int = 30,
        current_window_hours: int = 24,
        drift_threshold: float = 0.1,
    ):
        """
        Initialize drift detector.

        Args:
            baseline_window_days: Days for baseline window
            current_window_hours: Hours for current window
            drift_threshold: Threshold for drift detection
        """
        self.baseline_window = timedelta(days=baseline_window_days)
        self.current_window = timedelta(hours=current_window_hours)
        self.drift_threshold = drift_threshold

        self._baseline_data: Dict[str, List[float]] = {}
        self._current_data: Dict[str, List[float]] = {}

    def set_baseline(self, feature_name: str, values: List[float]):
        """
        Set baseline distribution for a feature.

        Args:
            feature_name: Name of feature
            values: Baseline values
        """
        self._baseline_data[feature_name] = values

    def add_observation(self, feature_name: str, value: float):
        """
        Add observation to current window.

        Args:
            feature_name: Name of feature
            value: Observed value
        """
        if feature_name not in self._current_data:
            self._current_data[feature_name] = []
        self._current_data[feature_name].append(value)

    def detect_drift(
        self,
        feature_name: str,
        drift_type: DriftType = DriftType.FEATURE_DRIFT,
    ) -> DriftResult:
        """
        Detect drift for a feature.

        Uses Population Stability Index (PSI) approximation.

        Args:
            feature_name: Feature to check
            drift_type: Type of drift to detect

        Returns:
            DriftResult with analysis
        """
        if feature_name not in self._baseline_data:
            raise KeyError(f"No baseline for feature: {feature_name}")

        baseline = self._baseline_data[feature_name]
        current = self._current_data.get(feature_name, [])

        if len(current) < 10:
            return DriftResult(
                feature_name=feature_name,
                drift_type=drift_type,
                timestamp=datetime.now(timezone.utc),
                baseline_period=(
                    datetime.now(timezone.utc) - self.baseline_window,
                    datetime.now(timezone.utc),
                ),
                current_period=(
                    datetime.now(timezone.utc) - self.current_window,
                    datetime.now(timezone.utc),
                ),
                drift_score=0.0,
                drift_detected=False,
                threshold=self.drift_threshold,
                statistical_test="PSI (insufficient data)",
                baseline_sample_count=len(baseline),
                current_sample_count=len(current),
            )

        # Calculate PSI-like score (simplified)
        baseline_mean = statistics.mean(baseline)
        baseline_std = statistics.stdev(baseline) if len(baseline) > 1 else 1.0
        current_mean = statistics.mean(current)
        current_std = statistics.stdev(current) if len(current) > 1 else 1.0

        # Normalized mean shift
        if baseline_std > 0:
            drift_score = abs(current_mean - baseline_mean) / baseline_std
        else:
            drift_score = abs(current_mean - baseline_mean)

        drift_detected = drift_score > self.drift_threshold

        return DriftResult(
            feature_name=feature_name,
            drift_type=drift_type,
            timestamp=datetime.now(timezone.utc),
            baseline_period=(
                datetime.now(timezone.utc) - self.baseline_window,
                datetime.now(timezone.utc),
            ),
            current_period=(
                datetime.now(timezone.utc) - self.current_window,
                datetime.now(timezone.utc),
            ),
            drift_score=drift_score,
            drift_detected=drift_detected,
            threshold=self.drift_threshold,
            statistical_test="Normalized Mean Shift",
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            current_mean=current_mean,
            current_std=current_std,
            baseline_sample_count=len(baseline),
            current_sample_count=len(current),
        )

    def detect_all_drift(self) -> List[DriftResult]:
        """Detect drift for all features."""
        results = []
        for feature_name in self._baseline_data.keys():
            result = self.detect_drift(feature_name)
            results.append(result)
        return results

    def reset_current_window(self):
        """Reset current observation window."""
        self._current_data = {}


class ModelMonitor:
    """
    Comprehensive model monitor.

    Orchestrates performance tracking, drift detection, and alerting.
    """

    def __init__(
        self,
        model_id: str,
        model_version: str,
        thresholds: Optional[List[AlertThreshold]] = None,
    ):
        """
        Initialize model monitor.

        Args:
            model_id: Model identifier
            model_version: Model version
            thresholds: Alert thresholds
        """
        self.model_id = model_id
        self.model_version = model_version

        self.performance_tracker = PerformanceTracker(model_id, model_version)
        self.drift_detector = DriftDetector()

        self._thresholds = thresholds or self._default_thresholds()
        self._alerts: List[MonitoringAlert] = []
        self._alert_cooldowns: Dict[str, datetime] = {}

    def _default_thresholds(self) -> List[AlertThreshold]:
        """Get default alert thresholds."""
        return [
            AlertThreshold(
                metric_name="accuracy",
                warning_threshold=0.85,
                critical_threshold=0.75,
                direction="below",
            ),
            AlertThreshold(
                metric_name="false_positive_rate",
                warning_threshold=0.10,
                critical_threshold=0.20,
                direction="above",
            ),
            AlertThreshold(
                metric_name="latency_p95_ms",
                warning_threshold=100,
                critical_threshold=200,
                direction="above",
            ),
            AlertThreshold(
                metric_name="drift_score",
                warning_threshold=0.15,
                critical_threshold=0.25,
                direction="above",
            ),
        ]

    def record_prediction(
        self,
        prediction: float,
        features: Dict[str, float],
        ground_truth: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ):
        """
        Record a prediction and check for issues.

        Args:
            prediction: Model prediction
            features: Input features
            ground_truth: Actual label (if available)
            latency_ms: Prediction latency
        """
        # Record for performance tracking
        self.performance_tracker.record_prediction(
            prediction=prediction,
            ground_truth=ground_truth,
            features=features,
            latency_ms=latency_ms,
        )

        # Add features to drift detector
        for feature_name, value in features.items():
            self.drift_detector.add_observation(feature_name, value)

        # Add prediction to drift detector
        self.drift_detector.add_observation("prediction", prediction)

    def check_alerts(self) -> List[MonitoringAlert]:
        """
        Check for alert conditions.

        Returns:
            List of new alerts generated
        """
        new_alerts = []
        snapshots = self.performance_tracker.get_snapshots(
            start_time=datetime.now(timezone.utc) - timedelta(hours=1)
        )

        if not snapshots:
            return new_alerts

        latest = snapshots[-1]

        for threshold in self._thresholds:
            value = getattr(latest, threshold.metric_name, None)
            if value is None:
                continue

            severity = threshold.check_threshold(value)
            if severity and self._can_alert(threshold.metric_name):
                alert = self._create_alert(
                    alert_type="threshold_breach",
                    severity=severity,
                    message=(
                        f"{threshold.metric_name} is {value:.3f}, "
                        f"{'above' if threshold.direction == 'above' else 'below'} "
                        f"threshold {threshold.warning_threshold}"
                    ),
                    metric_name=threshold.metric_name,
                    metric_value=value,
                    threshold_value=threshold.warning_threshold,
                )
                new_alerts.append(alert)
                self._alert_cooldowns[threshold.metric_name] = (
                    datetime.now(timezone.utc)
                )

        return new_alerts

    def _can_alert(self, metric_name: str) -> bool:
        """Check if alert cooldown has passed."""
        if metric_name not in self._alert_cooldowns:
            return True
        elapsed = datetime.now(timezone.utc) - self._alert_cooldowns[metric_name]
        return elapsed > timedelta(minutes=30)

    def _create_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold_value: float,
    ) -> MonitoringAlert:
        """Create a new alert."""
        import uuid

        alert = MonitoringAlert(
            alert_id=f"ALERT-{uuid.uuid4().hex[:8].upper()}",
            model_id=self.model_id,
            model_version=self.model_version,
            alert_type=alert_type,
            severity=severity,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold_value=threshold_value,
        )

        self._alerts.append(alert)
        logger.warning(f"Alert generated: {alert.alert_id} - {message}")

        return alert

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ):
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now(timezone.utc)
                return alert
        raise KeyError(f"Alert not found: {alert_id}")

    def resolve_alert(
        self,
        alert_id: str,
        resolution_notes: str = "",
    ):
        """Resolve an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now(timezone.utc)
                alert.resolution_notes = resolution_notes
                return alert
        raise KeyError(f"Alert not found: {alert_id}")

    def generate_report(
        self,
        period_start: datetime,
        period_end: datetime,
    ) -> MonitoringReport:
        """
        Generate comprehensive monitoring report.

        Args:
            period_start: Report period start
            period_end: Report period end

        Returns:
            MonitoringReport
        """
        import uuid

        # Get snapshots
        snapshots = self.performance_tracker.get_snapshots(
            start_time=period_start,
            end_time=period_end,
        )

        # Calculate averages
        accuracies = [s.accuracy for s in snapshots if s.accuracy is not None]
        precisions = [s.precision for s in snapshots if s.precision is not None]
        recalls = [s.recall for s in snapshots if s.recall is not None]
        latencies = [s.latency_p50_ms for s in snapshots if s.latency_p50_ms is not None]

        avg_accuracy = statistics.mean(accuracies) if accuracies else None
        avg_precision = statistics.mean(precisions) if precisions else None
        avg_recall = statistics.mean(recalls) if recalls else None
        avg_latency = statistics.mean(latencies) if latencies else None

        # Get drift results
        drift_results = self.drift_detector.detect_all_drift()
        features_with_drift = sum(1 for d in drift_results if d.drift_detected)

        # Get alerts
        period_alerts = [
            a for a in self._alerts
            if period_start <= a.created_at <= period_end
        ]
        active_alerts = [a for a in self._alerts if a.status == AlertStatus.ACTIVE]
        critical_alerts = [
            a for a in active_alerts if a.severity == AlertSeverity.CRITICAL
        ]

        # Calculate health score
        health_score = 100.0
        health_factors = {}

        if avg_accuracy is not None:
            accuracy_factor = min(avg_accuracy / 0.90, 1.0) * 30  # 30% weight
            health_factors["accuracy"] = accuracy_factor
            health_score = accuracy_factor

            if avg_precision is not None:
                precision_factor = min(avg_precision / 0.85, 1.0) * 20
                health_factors["precision"] = precision_factor
                health_score += precision_factor

            if avg_recall is not None:
                recall_factor = min(avg_recall / 0.85, 1.0) * 20
                health_factors["recall"] = recall_factor
                health_score += recall_factor

        # Drift penalty
        drift_penalty = min(features_with_drift * 5, 15)
        health_factors["drift_penalty"] = -drift_penalty
        health_score -= drift_penalty

        # Alert penalty
        alert_penalty = len(critical_alerts) * 10 + len(active_alerts) * 2
        health_factors["alert_penalty"] = -min(alert_penalty, 20)
        health_score -= min(alert_penalty, 20)

        health_score = max(0, min(100, health_score))

        return MonitoringReport(
            report_id=f"MON-{uuid.uuid4().hex[:8].upper()}",
            model_id=self.model_id,
            model_version=self.model_version,
            generated_at=datetime.now(timezone.utc),
            period_start=period_start,
            period_end=period_end,
            performance_snapshots=snapshots,
            avg_accuracy=avg_accuracy,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            drift_results=drift_results,
            features_with_drift=features_with_drift,
            alerts_generated=len(period_alerts),
            active_alerts=len(active_alerts),
            critical_alerts=len(critical_alerts),
            total_predictions=sum(s.sample_count for s in snapshots),
            avg_latency_ms=avg_latency,
            health_score=health_score,
            health_factors=health_factors,
        )
