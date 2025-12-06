"""
Model Performance Monitoring - Real-time Model Performance Monitoring.

This module provides comprehensive model performance monitoring for GreenLang
Process Heat agents, including prediction logging, metric tracking, anomaly
detection, and Prometheus-compatible metrics export.

Example:
    >>> from greenlang.ml.mlops.monitoring import ModelMonitor
    >>> monitor = ModelMonitor()
    >>> monitor.log_prediction("heat_model", features, prediction, latency_ms=15.2)
    >>> metrics = monitor.get_metrics("heat_model", window="24h")
    >>> alert = monitor.check_degradation("heat_model")
"""

import hashlib
import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import MLOpsConfig, get_config
from .schemas import (
    Alert,
    AlertLevel,
    PerformanceMetrics,
    PredictionLog,
    PrometheusMetric,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Time Window Utilities
# =============================================================================

def _parse_window(window: str) -> timedelta:
    """
    Parse time window string to timedelta.

    Args:
        window: Time window string (e.g., "1h", "24h", "7d").

    Returns:
        Corresponding timedelta.
    """
    value = int(window[:-1])
    unit = window[-1]

    if unit == "m":
        return timedelta(minutes=value)
    elif unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)
    elif unit == "w":
        return timedelta(weeks=value)
    else:
        raise ValueError(f"Invalid window unit: {unit}")


def _calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile of a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    return float(sorted_values[min(index, len(sorted_values) - 1)])


# =============================================================================
# Model Monitor Implementation
# =============================================================================

class ModelMonitor:
    """
    Real-time model performance monitoring.

    This class provides comprehensive monitoring capabilities including:
    - Prediction logging with features and latency
    - Actual value recording for delayed feedback
    - Rolling window metrics (1h, 24h, 7d)
    - Anomaly detection on metrics
    - Performance degradation alerts
    - Prometheus-compatible metrics export

    Attributes:
        config: MLOps configuration
        storage_path: Path for monitoring data storage
        predictions: In-memory prediction logs per model

    Example:
        >>> monitor = ModelMonitor()
        >>> pred_id = monitor.log_prediction(
        ...     model_name="heat_predictor",
        ...     features={"temp": 100, "pressure": 2.5},
        ...     prediction=85.3,
        ...     latency_ms=12.5
        ... )
        >>> # Later, when actual value is known
        >>> monitor.log_actual("heat_predictor", pred_id, actual=84.1)
        >>> # Get metrics
        >>> metrics = monitor.get_metrics("heat_predictor", window="24h")
        >>> print(f"MAE: {metrics.mae}, RMSE: {metrics.rmse}")
    """

    def __init__(self, config: Optional[MLOpsConfig] = None):
        """
        Initialize ModelMonitor.

        Args:
            config: MLOps configuration. If None, uses default configuration.
        """
        self.config = config or get_config()
        self.storage_path = Path(self.config.monitoring.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()

        # In-memory prediction logs (limited by retention)
        self._predictions: Dict[str, Deque[PredictionLog]] = {}
        self._max_memory_logs = 100000

        # Cached metrics
        self._metrics_cache: Dict[str, Dict[str, PerformanceMetrics]] = {}
        self._cache_ttl_seconds = 60

        # Alert tracking
        self._active_alerts: Dict[str, List[Alert]] = {}
        self._alert_cooldowns: Dict[str, datetime] = {}

        # Baseline metrics for degradation detection
        self._baseline_metrics: Dict[str, Dict[str, float]] = {}

        # Prometheus metrics
        self._prometheus_metrics: List[PrometheusMetric] = []

        logger.info("ModelMonitor initialized")

    def _generate_prediction_id(self) -> str:
        """Generate unique prediction identifier."""
        timestamp = datetime.utcnow().isoformat()
        random_suffix = np.random.randint(0, 999999)
        return hashlib.sha256(f"{timestamp}:{random_suffix}".encode()).hexdigest()[:16]

    def _hash_features(self, features: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of features."""
        features_str = json.dumps(features, sort_keys=True, default=str)
        return hashlib.sha256(features_str.encode()).hexdigest()

    def log_prediction(
        self,
        model_name: str,
        features: Dict[str, Any],
        prediction: float,
        latency_ms: float,
        model_version: str = "unknown",
        confidence: Optional[float] = None,
    ) -> str:
        """
        Log a model prediction.

        Args:
            model_name: Name of the model.
            features: Input features dictionary.
            prediction: Model prediction value.
            latency_ms: Prediction latency in milliseconds.
            model_version: Version of the model.
            confidence: Prediction confidence (optional).

        Returns:
            Unique prediction identifier.
        """
        prediction_id = self._generate_prediction_id()

        # Optionally truncate features for storage
        stored_features = features
        if self.config.monitoring.log_features:
            features_str = json.dumps(features, default=str)
            if len(features_str) > self.config.monitoring.max_feature_log_size:
                stored_features = {"_truncated": True, "_size": len(features_str)}

        log_entry = PredictionLog(
            prediction_id=prediction_id,
            model_name=model_name,
            model_version=model_version,
            features=stored_features if self.config.monitoring.log_features else {},
            features_hash=self._hash_features(features),
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
        )

        with self._lock:
            if model_name not in self._predictions:
                self._predictions[model_name] = deque(maxlen=self._max_memory_logs)

            self._predictions[model_name].append(log_entry)

        # Persist to disk periodically (every 100 predictions)
        if len(self._predictions[model_name]) % 100 == 0:
            self._persist_predictions(model_name)

        logger.debug(f"Logged prediction {prediction_id} for {model_name}")

        return prediction_id

    def log_actual(
        self, model_name: str, prediction_id: str, actual: float
    ) -> bool:
        """
        Log the actual value for a previous prediction.

        Args:
            model_name: Name of the model.
            prediction_id: Prediction identifier.
            actual: Actual (ground truth) value.

        Returns:
            True if prediction was found and updated.
        """
        if model_name not in self._predictions:
            return False

        with self._lock:
            for log_entry in self._predictions[model_name]:
                if log_entry.prediction_id == prediction_id:
                    log_entry.actual = actual
                    log_entry.actual_timestamp = datetime.utcnow()
                    logger.debug(f"Logged actual for {prediction_id}: {actual}")
                    return True

        return False

    def _persist_predictions(self, model_name: str) -> None:
        """Persist predictions to disk."""
        log_file = self.storage_path / f"predictions_{model_name}.jsonl"

        with self._lock:
            predictions = list(self._predictions.get(model_name, []))

        # Append to file (most recent batch)
        with open(log_file, "a") as f:
            for pred in predictions[-100:]:  # Last 100
                f.write(pred.json() + "\n")

    def get_metrics(
        self, model_name: str, window: str = "24h"
    ) -> PerformanceMetrics:
        """
        Get performance metrics for a model within a time window.

        Args:
            model_name: Name of the model.
            window: Time window (e.g., "1h", "24h", "7d").

        Returns:
            PerformanceMetrics with aggregated statistics.
        """
        # Check cache
        cache_key = f"{model_name}:{window}"
        if cache_key in self._metrics_cache:
            cached = self._metrics_cache[cache_key]
            cache_time = cached.get("_cached_at", datetime.min)
            if (datetime.utcnow() - cache_time).total_seconds() < self._cache_ttl_seconds:
                return cached["metrics"]

        # Calculate metrics
        window_delta = _parse_window(window)
        cutoff = datetime.utcnow() - window_delta

        predictions = self._predictions.get(model_name, [])
        recent_predictions = [
            p for p in predictions if p.timestamp > cutoff
        ]

        if not recent_predictions:
            return PerformanceMetrics(
                model_name=model_name,
                model_version="unknown",
                window=window,
                start_time=cutoff,
                end_time=datetime.utcnow(),
                prediction_count=0,
                predictions_with_actuals=0,
                mae=None,
                rmse=None,
                mape=None,
                r2_score=None,
                mean_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                max_latency_ms=0.0,
                throughput_per_second=0.0,
                error_count=0,
                error_rate=0.0,
            )

        # Calculate latency metrics
        latencies = [p.latency_ms for p in recent_predictions]

        # Calculate prediction accuracy metrics (only for predictions with actuals)
        predictions_with_actuals = [
            p for p in recent_predictions if p.actual is not None
        ]

        mae = None
        rmse = None
        mape = None
        r2_score = None

        if predictions_with_actuals:
            predictions_arr = np.array([p.prediction for p in predictions_with_actuals])
            actuals_arr = np.array([p.actual for p in predictions_with_actuals])

            errors = actuals_arr - predictions_arr
            abs_errors = np.abs(errors)

            mae = float(np.mean(abs_errors))
            rmse = float(np.sqrt(np.mean(errors ** 2)))

            # MAPE (avoid division by zero)
            nonzero_mask = actuals_arr != 0
            if np.any(nonzero_mask):
                mape = float(
                    np.mean(np.abs(errors[nonzero_mask] / actuals_arr[nonzero_mask])) * 100
                )

            # R2 score
            ss_res = np.sum(errors ** 2)
            ss_tot = np.sum((actuals_arr - np.mean(actuals_arr)) ** 2)
            if ss_tot > 0:
                r2_score = float(1 - ss_res / ss_tot)

        # Calculate throughput
        if len(recent_predictions) >= 2:
            time_span = (
                recent_predictions[-1].timestamp - recent_predictions[0].timestamp
            ).total_seconds()
            throughput = len(recent_predictions) / max(time_span, 1)
        else:
            throughput = 0.0

        # Get model version from most recent prediction
        model_version = recent_predictions[-1].model_version

        metrics = PerformanceMetrics(
            model_name=model_name,
            model_version=model_version,
            window=window,
            start_time=cutoff,
            end_time=datetime.utcnow(),
            prediction_count=len(recent_predictions),
            predictions_with_actuals=len(predictions_with_actuals),
            mae=mae,
            rmse=rmse,
            mape=mape,
            r2_score=r2_score,
            mean_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=_calculate_percentile(latencies, 50),
            p95_latency_ms=_calculate_percentile(latencies, 95),
            p99_latency_ms=_calculate_percentile(latencies, 99),
            max_latency_ms=float(max(latencies)),
            throughput_per_second=throughput,
            error_count=0,  # Would need error tracking
            error_rate=0.0,
        )

        # Cache result
        self._metrics_cache[cache_key] = {
            "metrics": metrics,
            "_cached_at": datetime.utcnow(),
        }

        return metrics

    def set_baseline_metrics(
        self, model_name: str, metrics: Dict[str, float]
    ) -> None:
        """
        Set baseline metrics for degradation detection.

        Args:
            model_name: Name of the model.
            metrics: Baseline metrics dictionary.
        """
        self._baseline_metrics[model_name] = metrics
        logger.info(f"Set baseline metrics for {model_name}: {metrics}")

    def check_degradation(self, model_name: str) -> Optional[Alert]:
        """
        Check for model performance degradation.

        Args:
            model_name: Name of the model.

        Returns:
            Alert if degradation detected, None otherwise.
        """
        # Get current metrics
        metrics = self.get_metrics(model_name, window="1h")

        if metrics.prediction_count < 100:
            return None  # Not enough data

        alerts = []

        # Check latency thresholds
        if metrics.p95_latency_ms > self.config.monitoring.latency_p95_threshold_ms:
            alerts.append(
                self._create_alert(
                    model_name=model_name,
                    model_version=metrics.model_version,
                    level=AlertLevel.WARNING,
                    alert_type="latency_degradation",
                    message=(
                        f"P95 latency {metrics.p95_latency_ms:.1f}ms exceeds "
                        f"threshold {self.config.monitoring.latency_p95_threshold_ms}ms"
                    ),
                    metric_name="p95_latency_ms",
                    metric_value=metrics.p95_latency_ms,
                    threshold=self.config.monitoring.latency_p95_threshold_ms,
                )
            )

        if metrics.p99_latency_ms > self.config.monitoring.latency_p99_threshold_ms:
            alerts.append(
                self._create_alert(
                    model_name=model_name,
                    model_version=metrics.model_version,
                    level=AlertLevel.ERROR,
                    alert_type="latency_critical",
                    message=(
                        f"P99 latency {metrics.p99_latency_ms:.1f}ms exceeds "
                        f"threshold {self.config.monitoring.latency_p99_threshold_ms}ms"
                    ),
                    metric_name="p99_latency_ms",
                    metric_value=metrics.p99_latency_ms,
                    threshold=self.config.monitoring.latency_p99_threshold_ms,
                )
            )

        # Check accuracy degradation against baseline
        if model_name in self._baseline_metrics:
            baseline = self._baseline_metrics[model_name]

            if metrics.mae is not None and "mae" in baseline:
                degradation = (metrics.mae - baseline["mae"]) / max(
                    baseline["mae"], 1e-10
                )
                if degradation > self.config.monitoring.mae_degradation_threshold:
                    alerts.append(
                        self._create_alert(
                            model_name=model_name,
                            model_version=metrics.model_version,
                            level=AlertLevel.WARNING,
                            alert_type="accuracy_degradation",
                            message=(
                                f"MAE increased by {degradation*100:.1f}% from baseline"
                            ),
                            metric_name="mae",
                            metric_value=metrics.mae,
                            threshold=baseline["mae"] * (
                                1 + self.config.monitoring.mae_degradation_threshold
                            ),
                        )
                    )

            if metrics.rmse is not None and "rmse" in baseline:
                degradation = (metrics.rmse - baseline["rmse"]) / max(
                    baseline["rmse"], 1e-10
                )
                if degradation > self.config.monitoring.rmse_degradation_threshold:
                    alerts.append(
                        self._create_alert(
                            model_name=model_name,
                            model_version=metrics.model_version,
                            level=AlertLevel.WARNING,
                            alert_type="accuracy_degradation",
                            message=(
                                f"RMSE increased by {degradation*100:.1f}% from baseline"
                            ),
                            metric_name="rmse",
                            metric_value=metrics.rmse,
                            threshold=baseline["rmse"] * (
                                1 + self.config.monitoring.rmse_degradation_threshold
                            ),
                        )
                    )

        # Return most severe alert
        if alerts:
            # Sort by severity
            severity_order = {
                AlertLevel.CRITICAL: 0,
                AlertLevel.ERROR: 1,
                AlertLevel.WARNING: 2,
                AlertLevel.INFO: 3,
            }
            alerts.sort(key=lambda a: severity_order.get(a.level, 99))
            return alerts[0]

        return None

    def _create_alert(
        self,
        model_name: str,
        model_version: str,
        level: AlertLevel,
        alert_type: str,
        message: str,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> Alert:
        """Create an alert with cooldown checking."""
        # Check cooldown
        cooldown_key = f"{model_name}:{alert_type}"
        if cooldown_key in self._alert_cooldowns:
            last_alert = self._alert_cooldowns[cooldown_key]
            cooldown_minutes = self.config.alert.cooldown_minutes
            if (datetime.utcnow() - last_alert).total_seconds() < cooldown_minutes * 60:
                return None

        alert = Alert(
            alert_id=hashlib.sha256(
                f"{model_name}:{alert_type}:{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()[:16],
            model_name=model_name,
            model_version=model_version,
            level=level,
            alert_type=alert_type,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            timestamp=datetime.utcnow(),
        )

        # Update cooldown
        self._alert_cooldowns[cooldown_key] = datetime.utcnow()

        # Store alert
        if model_name not in self._active_alerts:
            self._active_alerts[model_name] = []
        self._active_alerts[model_name].append(alert)

        # Persist alert
        self._persist_alert(alert)

        logger.warning(f"Alert created: [{level.value}] {model_name} - {message}")

        return alert

    def _persist_alert(self, alert: Alert) -> None:
        """Persist alert to storage."""
        alert_file = self.storage_path / f"alerts_{alert.model_name}.jsonl"
        with open(alert_file, "a") as f:
            f.write(alert.json() + "\n")

    def get_active_alerts(
        self, model_name: Optional[str] = None
    ) -> List[Alert]:
        """
        Get active (unresolved) alerts.

        Args:
            model_name: Filter by model name (optional).

        Returns:
            List of active Alert objects.
        """
        alerts = []

        if model_name:
            alerts = [
                a
                for a in self._active_alerts.get(model_name, [])
                if not a.resolved
            ]
        else:
            for model_alerts in self._active_alerts.values():
                alerts.extend([a for a in model_alerts if not a.resolved])

        return alerts

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert identifier.
            acknowledged_by: User who acknowledged.

        Returns:
            True if alert was found and acknowledged.
        """
        for model_alerts in self._active_alerts.values():
            for alert in model_alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_at = datetime.utcnow()
                    return True

        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert identifier.

        Returns:
            True if alert was found and resolved.
        """
        for model_alerts in self._active_alerts.values():
            for alert in model_alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    return True

        return False

    def export_prometheus_metrics(self, model_name: str) -> List[PrometheusMetric]:
        """
        Export metrics in Prometheus format.

        Args:
            model_name: Name of the model.

        Returns:
            List of PrometheusMetric objects.
        """
        if not self.config.monitoring.prometheus_enabled:
            return []

        metrics = self.get_metrics(model_name, window="1h")
        prefix = self.config.monitoring.prometheus_prefix
        timestamp_ms = int(datetime.utcnow().timestamp() * 1000)

        prometheus_metrics = []

        # Prediction count
        prometheus_metrics.append(
            PrometheusMetric(
                name=f"{prefix}_predictions_total",
                help_text="Total number of predictions",
                type="counter",
                labels={"model": model_name, "version": metrics.model_version},
                value=float(metrics.prediction_count),
                timestamp_ms=timestamp_ms,
            )
        )

        # Latency metrics
        prometheus_metrics.append(
            PrometheusMetric(
                name=f"{prefix}_latency_mean_ms",
                help_text="Mean prediction latency in milliseconds",
                type="gauge",
                labels={"model": model_name},
                value=metrics.mean_latency_ms,
                timestamp_ms=timestamp_ms,
            )
        )

        prometheus_metrics.append(
            PrometheusMetric(
                name=f"{prefix}_latency_p95_ms",
                help_text="P95 prediction latency in milliseconds",
                type="gauge",
                labels={"model": model_name},
                value=metrics.p95_latency_ms,
                timestamp_ms=timestamp_ms,
            )
        )

        prometheus_metrics.append(
            PrometheusMetric(
                name=f"{prefix}_latency_p99_ms",
                help_text="P99 prediction latency in milliseconds",
                type="gauge",
                labels={"model": model_name},
                value=metrics.p99_latency_ms,
                timestamp_ms=timestamp_ms,
            )
        )

        # Accuracy metrics (if available)
        if metrics.mae is not None:
            prometheus_metrics.append(
                PrometheusMetric(
                    name=f"{prefix}_mae",
                    help_text="Mean Absolute Error",
                    type="gauge",
                    labels={"model": model_name},
                    value=metrics.mae,
                    timestamp_ms=timestamp_ms,
                )
            )

        if metrics.rmse is not None:
            prometheus_metrics.append(
                PrometheusMetric(
                    name=f"{prefix}_rmse",
                    help_text="Root Mean Square Error",
                    type="gauge",
                    labels={"model": model_name},
                    value=metrics.rmse,
                    timestamp_ms=timestamp_ms,
                )
            )

        # Throughput
        prometheus_metrics.append(
            PrometheusMetric(
                name=f"{prefix}_throughput_per_second",
                help_text="Predictions per second",
                type="gauge",
                labels={"model": model_name},
                value=metrics.throughput_per_second,
                timestamp_ms=timestamp_ms,
            )
        )

        return prometheus_metrics

    def get_prometheus_text(self, model_name: str) -> str:
        """
        Get Prometheus metrics in text exposition format.

        Args:
            model_name: Name of the model.

        Returns:
            Prometheus text format string.
        """
        metrics = self.export_prometheus_metrics(model_name)
        lines = []

        for metric in metrics:
            # Help line
            lines.append(f"# HELP {metric.name} {metric.help_text}")
            # Type line
            lines.append(f"# TYPE {metric.name} {metric.type}")
            # Metric line
            labels_str = ",".join(
                f'{k}="{v}"' for k, v in metric.labels.items()
            )
            if metric.timestamp_ms:
                lines.append(f"{metric.name}{{{labels_str}}} {metric.value} {metric.timestamp_ms}")
            else:
                lines.append(f"{metric.name}{{{labels_str}}} {metric.value}")

        return "\n".join(lines)

    def get_dashboard_data(self, model_name: str) -> Dict[str, Any]:
        """
        Get data for monitoring dashboard.

        Args:
            model_name: Name of the model.

        Returns:
            Dictionary with dashboard-ready data.
        """
        # Get metrics for different windows
        metrics_1h = self.get_metrics(model_name, "1h")
        metrics_24h = self.get_metrics(model_name, "24h")
        metrics_7d = self.get_metrics(model_name, "7d")

        # Get active alerts
        alerts = self.get_active_alerts(model_name)

        return {
            "model_name": model_name,
            "updated_at": datetime.utcnow().isoformat(),
            "metrics": {
                "1h": {
                    "prediction_count": metrics_1h.prediction_count,
                    "mae": metrics_1h.mae,
                    "rmse": metrics_1h.rmse,
                    "mean_latency_ms": metrics_1h.mean_latency_ms,
                    "p95_latency_ms": metrics_1h.p95_latency_ms,
                    "throughput": metrics_1h.throughput_per_second,
                },
                "24h": {
                    "prediction_count": metrics_24h.prediction_count,
                    "mae": metrics_24h.mae,
                    "rmse": metrics_24h.rmse,
                    "mean_latency_ms": metrics_24h.mean_latency_ms,
                    "p95_latency_ms": metrics_24h.p95_latency_ms,
                    "throughput": metrics_24h.throughput_per_second,
                },
                "7d": {
                    "prediction_count": metrics_7d.prediction_count,
                    "mae": metrics_7d.mae,
                    "rmse": metrics_7d.rmse,
                    "mean_latency_ms": metrics_7d.mean_latency_ms,
                    "p95_latency_ms": metrics_7d.p95_latency_ms,
                    "throughput": metrics_7d.throughput_per_second,
                },
            },
            "alerts": {
                "active_count": len(alerts),
                "critical": len([a for a in alerts if a.level == AlertLevel.CRITICAL]),
                "error": len([a for a in alerts if a.level == AlertLevel.ERROR]),
                "warning": len([a for a in alerts if a.level == AlertLevel.WARNING]),
                "recent": [
                    {
                        "id": a.alert_id,
                        "level": a.level.value,
                        "message": a.message,
                        "timestamp": a.timestamp.isoformat(),
                    }
                    for a in sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:5]
                ],
            },
            "baseline": self._baseline_metrics.get(model_name, {}),
        }

    def cleanup_old_logs(self, days: Optional[int] = None) -> int:
        """
        Clean up old prediction logs.

        Args:
            days: Days to retain. Defaults to config value.

        Returns:
            Number of logs cleaned up.
        """
        days = days or self.config.monitoring.retention_days
        cutoff = datetime.utcnow() - timedelta(days=days)
        cleaned = 0

        with self._lock:
            for model_name, predictions in self._predictions.items():
                original_count = len(predictions)

                # Filter to keep only recent predictions
                recent = [p for p in predictions if p.timestamp > cutoff]
                self._predictions[model_name] = deque(
                    recent, maxlen=self._max_memory_logs
                )

                cleaned += original_count - len(recent)

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old prediction logs")

        return cleaned
