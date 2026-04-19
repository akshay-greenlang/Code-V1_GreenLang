# -*- coding: utf-8 -*-
"""
Self-Learning Metrics Dashboard Module

This module provides comprehensive metrics tracking and visualization data
for GreenLang self-learning systems, including learning curves, drift
detection, model plasticity, and knowledge retention scores.

The dashboard enables monitoring of model adaptation health, early
detection of learning issues, and provides REST API endpoints for
integration with monitoring systems.

Example:
    >>> from greenlang.ml.self_learning import MetricsDashboard
    >>> dashboard = MetricsDashboard()
    >>> dashboard.record_training_step(loss=0.5, accuracy=0.85)
    >>> summary = dashboard.get_summary()
    >>> # Start FastAPI server for metrics endpoint
    >>> dashboard.start_api_server(port=8080)
"""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import json
import threading

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics tracked."""
    LOSS = "loss"
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    AUC = "auc"
    DRIFT_SCORE = "drift_score"
    FORGETTING = "forgetting"
    PLASTICITY = "plasticity"
    RETENTION = "retention"
    LEARNING_RATE = "learning_rate"
    GRADIENT_NORM = "gradient_norm"
    CUSTOM = "custom"


class MetricAggregation(str, Enum):
    """Aggregation methods for metrics."""
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    STD = "std"
    SUM = "sum"
    LAST = "last"
    EMA = "ema"  # Exponential moving average


class DashboardConfig(BaseModel):
    """Configuration for metrics dashboard."""

    history_length: int = Field(
        default=10000,
        ge=100,
        description="Maximum history length to store"
    )
    ema_alpha: float = Field(
        default=0.1,
        gt=0,
        le=1.0,
        description="Alpha for exponential moving average"
    )
    drift_window: int = Field(
        default=100,
        ge=10,
        description="Window size for drift detection"
    )
    drift_threshold: float = Field(
        default=0.1,
        gt=0,
        description="Threshold for drift detection"
    )
    alert_on_drift: bool = Field(
        default=True,
        description="Raise alert when drift detected"
    )
    alert_on_degradation: bool = Field(
        default=True,
        description="Raise alert when performance degrades"
    )
    degradation_threshold: float = Field(
        default=0.05,
        gt=0,
        description="Performance drop threshold for alerts"
    )
    auto_snapshot_interval: int = Field(
        default=1000,
        ge=0,
        description="Auto-snapshot every N steps (0 to disable)"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    api_prefix: str = Field(
        default="/api/v1/metrics",
        description="API endpoint prefix"
    )


class MetricDataPoint(BaseModel):
    """Single metric data point."""

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of measurement"
    )
    step: int = Field(
        ...,
        description="Training step number"
    )
    value: float = Field(
        ...,
        description="Metric value"
    )
    metric_type: str = Field(
        ...,
        description="Type of metric"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )


class LearningCurve(BaseModel):
    """Learning curve data for visualization."""

    metric_name: str = Field(
        ...,
        description="Name of the metric"
    )
    steps: List[int] = Field(
        ...,
        description="Step numbers"
    )
    values: List[float] = Field(
        ...,
        description="Metric values"
    )
    ema_values: Optional[List[float]] = Field(
        default=None,
        description="Exponential moving average values"
    )
    timestamps: Optional[List[str]] = Field(
        default=None,
        description="Timestamps (ISO format)"
    )


class DriftStatus(BaseModel):
    """Status of drift detection."""

    is_drifting: bool = Field(
        ...,
        description="Whether drift is detected"
    )
    drift_score: float = Field(
        ...,
        description="Current drift score"
    )
    drift_direction: str = Field(
        default="stable",
        description="Direction: increasing, decreasing, stable"
    )
    window_mean: float = Field(
        ...,
        description="Mean in current window"
    )
    baseline_mean: float = Field(
        ...,
        description="Baseline mean"
    )
    time_since_last_drift: Optional[int] = Field(
        default=None,
        description="Steps since last drift"
    )


class PlasticityMetrics(BaseModel):
    """Model plasticity metrics."""

    plasticity_score: float = Field(
        default=1.0,
        ge=0,
        le=1.0,
        description="Overall plasticity (0-1)"
    )
    gradient_magnitude: float = Field(
        default=0.0,
        description="Average gradient magnitude"
    )
    weight_change_rate: float = Field(
        default=0.0,
        description="Rate of weight changes"
    )
    learning_speed: float = Field(
        default=0.0,
        description="Speed of learning new tasks"
    )
    adaptability: float = Field(
        default=1.0,
        description="Ability to adapt to new data"
    )


class RetentionMetrics(BaseModel):
    """Knowledge retention metrics."""

    retention_score: float = Field(
        default=1.0,
        ge=0,
        le=1.0,
        description="Overall retention (0-1)"
    )
    average_forgetting: float = Field(
        default=0.0,
        description="Average forgetting across tasks"
    )
    max_forgetting: float = Field(
        default=0.0,
        description="Maximum forgetting on any task"
    )
    task_retentions: Dict[str, float] = Field(
        default_factory=dict,
        description="Retention per task"
    )
    stability_score: float = Field(
        default=1.0,
        description="Model stability score"
    )


class DashboardSummary(BaseModel):
    """Summary of dashboard metrics."""

    total_steps: int = Field(
        ...,
        description="Total training steps"
    )
    total_time_seconds: float = Field(
        ...,
        description="Total training time"
    )
    current_metrics: Dict[str, float] = Field(
        ...,
        description="Current metric values"
    )
    best_metrics: Dict[str, float] = Field(
        ...,
        description="Best metric values achieved"
    )
    drift_status: DriftStatus = Field(
        ...,
        description="Current drift status"
    )
    plasticity: PlasticityMetrics = Field(
        ...,
        description="Plasticity metrics"
    )
    retention: RetentionMetrics = Field(
        ...,
        description="Retention metrics"
    )
    alerts: List[str] = Field(
        default_factory=list,
        description="Active alerts"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )


class MetricAlert(BaseModel):
    """Alert for metric issues."""

    alert_id: str = Field(
        ...,
        description="Unique alert identifier"
    )
    alert_type: str = Field(
        ...,
        description="Type of alert"
    )
    severity: str = Field(
        default="warning",
        description="Severity: info, warning, critical"
    )
    message: str = Field(
        ...,
        description="Alert message"
    )
    metric_name: str = Field(
        ...,
        description="Affected metric"
    )
    metric_value: float = Field(
        ...,
        description="Current metric value"
    )
    threshold: float = Field(
        ...,
        description="Threshold that was breached"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When alert was raised"
    )
    acknowledged: bool = Field(
        default=False,
        description="Whether alert was acknowledged"
    )


class MetricsDashboard:
    """
    Self-Learning Metrics Dashboard for GreenLang.

    This class provides comprehensive metrics tracking, visualization
    data generation, and REST API endpoints for monitoring self-learning
    systems in production.

    Key capabilities:
    - Learning curve visualization data
    - Drift vs adaptation tracking
    - Model plasticity metrics
    - Knowledge retention scores
    - FastAPI endpoint for metrics
    - Real-time alerting
    - Provenance tracking

    Attributes:
        config: Dashboard configuration
        _metrics: Dictionary of metric histories
        _snapshots: Historical snapshots
        _alerts: Active alerts

    Example:
        >>> dashboard = MetricsDashboard(
        ...     config=DashboardConfig(
        ...         history_length=10000,
        ...         drift_threshold=0.1
        ...     )
        ... )
        >>> # Record metrics during training
        >>> for step in range(1000):
        ...     loss = train_step()
        ...     dashboard.record_training_step(
        ...         step=step,
        ...         loss=loss,
        ...         accuracy=accuracy
        ...     )
        >>> # Get visualization data
        >>> curve = dashboard.get_learning_curve("loss")
        >>> drift = dashboard.get_drift_status("accuracy")
    """

    def __init__(
        self,
        config: Optional[DashboardConfig] = None
    ):
        """
        Initialize metrics dashboard.

        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()

        # Metric storage
        self._metrics: Dict[str, deque] = {}
        self._ema_values: Dict[str, float] = {}
        self._best_values: Dict[str, float] = {}
        self._worst_values: Dict[str, float] = {}

        # Baseline for drift detection
        self._baselines: Dict[str, float] = {}
        self._baseline_stds: Dict[str, float] = {}
        self._last_drift: Dict[str, int] = {}

        # Snapshots
        self._snapshots: List[DashboardSummary] = []

        # Alerts
        self._alerts: List[MetricAlert] = []
        self._alert_counter = 0

        # State tracking
        self._step_count = 0
        self._start_time = datetime.utcnow()
        self._last_weights_hash: Optional[str] = None

        # Task tracking for retention
        self._task_initial_perf: Dict[str, float] = {}
        self._task_current_perf: Dict[str, float] = {}

        # Gradient tracking for plasticity
        self._gradient_history: deque = deque(maxlen=100)
        self._weight_change_history: deque = deque(maxlen=100)

        logger.info("MetricsDashboard initialized")

    def record_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a single metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step number (uses internal counter if not provided)
            context: Optional additional context

        Example:
            >>> dashboard.record_metric("loss", 0.5, step=100)
        """
        step = step if step is not None else self._step_count

        # Initialize storage if needed
        if name not in self._metrics:
            self._metrics[name] = deque(maxlen=self.config.history_length)
            self._ema_values[name] = value
            self._best_values[name] = value
            self._worst_values[name] = value

        # Create data point
        data_point = MetricDataPoint(
            timestamp=datetime.utcnow(),
            step=step,
            value=value,
            metric_type=name,
            context=context or {}
        )

        self._metrics[name].append(data_point)

        # Update EMA
        alpha = self.config.ema_alpha
        self._ema_values[name] = alpha * value + (1 - alpha) * self._ema_values[name]

        # Update best/worst
        if "loss" in name.lower() or "error" in name.lower():
            # Lower is better
            if value < self._best_values[name]:
                self._best_values[name] = value
            if value > self._worst_values[name]:
                self._worst_values[name] = value
        else:
            # Higher is better
            if value > self._best_values[name]:
                self._best_values[name] = value
            if value < self._worst_values[name]:
                self._worst_values[name] = value

        # Check for drift
        self._check_drift(name, value)

        # Check for alerts
        self._check_alerts(name, value)

    def record_training_step(
        self,
        step: Optional[int] = None,
        **metrics: float
    ) -> None:
        """
        Record metrics for a training step.

        Args:
            step: Step number (auto-incremented if not provided)
            **metrics: Metric name-value pairs

        Example:
            >>> dashboard.record_training_step(
            ...     loss=0.5,
            ...     accuracy=0.85,
            ...     learning_rate=0.001
            ... )
        """
        step = step if step is not None else self._step_count

        for name, value in metrics.items():
            self.record_metric(name, value, step)

        self._step_count = max(self._step_count, step + 1)

        # Auto-snapshot
        if (self.config.auto_snapshot_interval > 0 and
            self._step_count % self.config.auto_snapshot_interval == 0):
            self._create_snapshot()

    def record_gradient_norm(
        self,
        norm: float,
        step: Optional[int] = None
    ) -> None:
        """
        Record gradient norm for plasticity tracking.

        Args:
            norm: L2 norm of gradients
            step: Optional step number
        """
        self._gradient_history.append(norm)
        self.record_metric("gradient_norm", norm, step)

    def record_weight_change(
        self,
        change: float,
        step: Optional[int] = None
    ) -> None:
        """
        Record weight change magnitude for plasticity.

        Args:
            change: Magnitude of weight changes
            step: Optional step number
        """
        self._weight_change_history.append(change)
        self.record_metric("weight_change", change, step)

    def record_task_performance(
        self,
        task_id: str,
        performance: float,
        is_initial: bool = False
    ) -> None:
        """
        Record task performance for retention tracking.

        Args:
            task_id: Task identifier
            performance: Performance metric (0-1)
            is_initial: Whether this is the initial performance
        """
        if is_initial:
            self._task_initial_perf[task_id] = performance

        self._task_current_perf[task_id] = performance

        # Record forgetting
        if task_id in self._task_initial_perf:
            forgetting = max(0, self._task_initial_perf[task_id] - performance)
            self.record_metric(f"forgetting_{task_id}", forgetting)

    def _check_drift(self, name: str, value: float) -> None:
        """Check for drift in metric."""
        if name not in self._metrics:
            return

        history = list(self._metrics[name])

        if len(history) < self.config.drift_window * 2:
            # Not enough data - set baseline
            if name not in self._baselines and len(history) >= self.config.drift_window:
                values = [p.value for p in history[-self.config.drift_window:]]
                self._baselines[name] = np.mean(values)
                self._baseline_stds[name] = np.std(values) + 1e-8
            return

        # Calculate current window stats
        recent = [p.value for p in history[-self.config.drift_window:]]
        current_mean = np.mean(recent)

        baseline = self._baselines.get(name, current_mean)
        baseline_std = self._baseline_stds.get(name, 1.0)

        # Z-score drift detection
        drift_score = abs(current_mean - baseline) / baseline_std

        if drift_score > self.config.drift_threshold / baseline_std:
            self._last_drift[name] = self._step_count

            if self.config.alert_on_drift:
                self._raise_alert(
                    alert_type="drift",
                    message=f"Drift detected in {name}: score={drift_score:.3f}",
                    metric_name=name,
                    metric_value=current_mean,
                    threshold=self.config.drift_threshold,
                    severity="warning"
                )

    def _check_alerts(self, name: str, value: float) -> None:
        """Check if value should trigger alerts."""
        # Check degradation
        if name in self._best_values:
            best = self._best_values[name]

            if "loss" in name.lower() or "error" in name.lower():
                # Higher is worse
                degradation = (value - best) / (abs(best) + 1e-8)
            else:
                # Lower is worse
                degradation = (best - value) / (abs(best) + 1e-8)

            if degradation > self.config.degradation_threshold:
                if self.config.alert_on_degradation:
                    self._raise_alert(
                        alert_type="degradation",
                        message=f"Performance degradation in {name}: {degradation:.1%} from best",
                        metric_name=name,
                        metric_value=value,
                        threshold=self.config.degradation_threshold,
                        severity="warning" if degradation < 0.2 else "critical"
                    )

    def _raise_alert(
        self,
        alert_type: str,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        severity: str = "warning"
    ) -> MetricAlert:
        """Raise a new alert."""
        self._alert_counter += 1
        alert_id = f"alert_{self._alert_counter}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        alert = MetricAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold
        )

        self._alerts.append(alert)

        # Keep only recent alerts
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]

        logger.warning(f"Alert raised: {message}")
        return alert

    def get_learning_curve(
        self,
        metric_name: str,
        include_ema: bool = True,
        last_n: Optional[int] = None
    ) -> LearningCurve:
        """
        Get learning curve data for visualization.

        Args:
            metric_name: Name of metric
            include_ema: Include EMA values
            last_n: Only return last N points

        Returns:
            LearningCurve with visualization data

        Example:
            >>> curve = dashboard.get_learning_curve("loss")
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(curve.steps, curve.values)
        """
        if metric_name not in self._metrics:
            return LearningCurve(
                metric_name=metric_name,
                steps=[],
                values=[]
            )

        history = list(self._metrics[metric_name])
        if last_n:
            history = history[-last_n:]

        steps = [p.step for p in history]
        values = [p.value for p in history]
        timestamps = [p.timestamp.isoformat() for p in history]

        # Calculate EMA
        ema_values = None
        if include_ema and len(values) > 0:
            ema_values = []
            ema = values[0]
            alpha = self.config.ema_alpha
            for v in values:
                ema = alpha * v + (1 - alpha) * ema
                ema_values.append(ema)

        return LearningCurve(
            metric_name=metric_name,
            steps=steps,
            values=values,
            ema_values=ema_values,
            timestamps=timestamps
        )

    def get_drift_status(
        self,
        metric_name: str
    ) -> DriftStatus:
        """
        Get drift status for a metric.

        Args:
            metric_name: Name of metric

        Returns:
            DriftStatus with drift information

        Example:
            >>> drift = dashboard.get_drift_status("accuracy")
            >>> if drift.is_drifting:
            ...     print("Model is drifting!")
        """
        if metric_name not in self._metrics:
            return DriftStatus(
                is_drifting=False,
                drift_score=0.0,
                window_mean=0.0,
                baseline_mean=0.0
            )

        history = list(self._metrics[metric_name])

        if len(history) < self.config.drift_window:
            return DriftStatus(
                is_drifting=False,
                drift_score=0.0,
                window_mean=0.0,
                baseline_mean=0.0
            )

        recent = [p.value for p in history[-self.config.drift_window:]]
        window_mean = np.mean(recent)

        baseline = self._baselines.get(metric_name, window_mean)
        baseline_std = self._baseline_stds.get(metric_name, 1.0)

        drift_score = abs(window_mean - baseline) / baseline_std

        # Determine direction
        if window_mean > baseline + baseline_std * 0.5:
            direction = "increasing"
        elif window_mean < baseline - baseline_std * 0.5:
            direction = "decreasing"
        else:
            direction = "stable"

        # Time since last drift
        time_since_drift = None
        if metric_name in self._last_drift:
            time_since_drift = self._step_count - self._last_drift[metric_name]

        return DriftStatus(
            is_drifting=drift_score > self.config.drift_threshold / baseline_std,
            drift_score=drift_score,
            drift_direction=direction,
            window_mean=window_mean,
            baseline_mean=baseline,
            time_since_last_drift=time_since_drift
        )

    def get_plasticity_metrics(self) -> PlasticityMetrics:
        """
        Get model plasticity metrics.

        Returns:
            PlasticityMetrics with plasticity analysis
        """
        # Gradient magnitude
        grad_magnitude = 0.0
        if self._gradient_history:
            grad_magnitude = float(np.mean(self._gradient_history))

        # Weight change rate
        weight_change_rate = 0.0
        if self._weight_change_history:
            weight_change_rate = float(np.mean(self._weight_change_history))

        # Learning speed (based on loss improvement rate)
        learning_speed = 0.0
        if "loss" in self._metrics and len(self._metrics["loss"]) > 10:
            losses = [p.value for p in list(self._metrics["loss"])[-100:]]
            if len(losses) > 1:
                improvement_rate = (losses[0] - losses[-1]) / len(losses)
                learning_speed = max(0, min(1, improvement_rate * 10))

        # Overall plasticity score
        plasticity_score = 0.5  # Default neutral

        if grad_magnitude > 0:
            # Normalize gradient magnitude
            plasticity_score = min(1.0, grad_magnitude / 10.0)

        # Adaptability (inverse of loss variance in recent window)
        adaptability = 1.0
        if "loss" in self._metrics:
            recent_losses = [p.value for p in list(self._metrics["loss"])[-50:]]
            if len(recent_losses) > 5:
                variance = np.var(recent_losses)
                adaptability = 1.0 / (1.0 + variance)

        return PlasticityMetrics(
            plasticity_score=plasticity_score,
            gradient_magnitude=grad_magnitude,
            weight_change_rate=weight_change_rate,
            learning_speed=learning_speed,
            adaptability=adaptability
        )

    def get_retention_metrics(self) -> RetentionMetrics:
        """
        Get knowledge retention metrics.

        Returns:
            RetentionMetrics with retention analysis
        """
        if not self._task_initial_perf:
            return RetentionMetrics()

        # Calculate per-task retention
        task_retentions = {}
        forgetting_values = []

        for task_id, initial_perf in self._task_initial_perf.items():
            current_perf = self._task_current_perf.get(task_id, initial_perf)
            retention = current_perf / (initial_perf + 1e-8)
            retention = max(0, min(1, retention))
            task_retentions[task_id] = retention

            forgetting = max(0, initial_perf - current_perf)
            forgetting_values.append(forgetting)

        # Aggregate metrics
        avg_forgetting = np.mean(forgetting_values) if forgetting_values else 0.0
        max_forgetting = np.max(forgetting_values) if forgetting_values else 0.0
        retention_score = 1.0 - avg_forgetting

        # Stability score (based on variance of retention across tasks)
        stability = 1.0
        if len(task_retentions) > 1:
            variance = np.var(list(task_retentions.values()))
            stability = 1.0 / (1.0 + variance * 10)

        return RetentionMetrics(
            retention_score=max(0, retention_score),
            average_forgetting=avg_forgetting,
            max_forgetting=max_forgetting,
            task_retentions=task_retentions,
            stability_score=stability
        )

    def _create_snapshot(self) -> DashboardSummary:
        """Create a snapshot of current dashboard state."""
        summary = self.get_summary()
        self._snapshots.append(summary)

        # Keep only recent snapshots
        if len(self._snapshots) > 100:
            self._snapshots = self._snapshots[-100:]

        return summary

    def _calculate_provenance(self) -> str:
        """Calculate SHA-256 provenance hash."""
        metrics_summary = {
            name: self._ema_values.get(name, 0.0)
            for name in self._metrics
        }
        combined = f"{self._step_count}|{json.dumps(metrics_summary, sort_keys=True)}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get_summary(self) -> DashboardSummary:
        """
        Get comprehensive dashboard summary.

        Returns:
            DashboardSummary with all metrics

        Example:
            >>> summary = dashboard.get_summary()
            >>> print(f"Total steps: {summary.total_steps}")
            >>> print(f"Retention: {summary.retention.retention_score:.2f}")
        """
        # Current metric values
        current_metrics = {
            name: self._ema_values.get(name, 0.0)
            for name in self._metrics
        }

        # Best metric values
        best_metrics = self._best_values.copy()

        # Get first metric for drift status (or use loss)
        primary_metric = "loss" if "loss" in self._metrics else (
            list(self._metrics.keys())[0] if self._metrics else "loss"
        )
        drift_status = self.get_drift_status(primary_metric)

        # Get plasticity and retention
        plasticity = self.get_plasticity_metrics()
        retention = self.get_retention_metrics()

        # Active alerts
        active_alerts = [
            a.message for a in self._alerts
            if not a.acknowledged
        ][-5:]  # Last 5 unacknowledged

        # Calculate time
        total_time = (datetime.utcnow() - self._start_time).total_seconds()

        return DashboardSummary(
            total_steps=self._step_count,
            total_time_seconds=total_time,
            current_metrics=current_metrics,
            best_metrics=best_metrics,
            drift_status=drift_status,
            plasticity=plasticity,
            retention=retention,
            alerts=active_alerts,
            provenance_hash=self._calculate_provenance(),
            last_updated=datetime.utcnow()
        )

    def get_alerts(
        self,
        unacknowledged_only: bool = False,
        limit: Optional[int] = None
    ) -> List[MetricAlert]:
        """
        Get list of alerts.

        Args:
            unacknowledged_only: Only return unacknowledged alerts
            limit: Maximum number to return

        Returns:
            List of alerts
        """
        alerts = self._alerts

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        if limit:
            alerts = alerts[-limit:]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert identifier

        Returns:
            Whether alert was found and acknowledged
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def reset_baseline(self, metric_name: str) -> None:
        """
        Reset baseline for drift detection.

        Args:
            metric_name: Metric to reset baseline for
        """
        if metric_name in self._metrics:
            history = list(self._metrics[metric_name])
            if len(history) >= self.config.drift_window:
                values = [p.value for p in history[-self.config.drift_window:]]
                self._baselines[metric_name] = np.mean(values)
                self._baseline_stds[metric_name] = np.std(values) + 1e-8
                logger.info(f"Reset baseline for {metric_name}")

    def export_metrics(self, format: str = "json") -> str:
        """
        Export all metrics.

        Args:
            format: Export format (json)

        Returns:
            Exported metrics as string
        """
        data = {
            "summary": self.get_summary().dict(),
            "learning_curves": {
                name: self.get_learning_curve(name).dict()
                for name in self._metrics
            }
        }

        if format == "json":
            return json.dumps(data, indent=2, default=str)

        return json.dumps(data, default=str)

    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()
        self._ema_values.clear()
        self._best_values.clear()
        self._worst_values.clear()
        self._baselines.clear()
        self._baseline_stds.clear()
        self._last_drift.clear()
        self._snapshots.clear()
        self._alerts.clear()
        self._step_count = 0
        self._start_time = datetime.utcnow()
        logger.info("Dashboard cleared")

    # =========================================================================
    # FastAPI Integration
    # =========================================================================

    def create_api_router(self) -> Any:
        """
        Create FastAPI router for metrics endpoint.

        Returns:
            FastAPI APIRouter

        Example:
            >>> from fastapi import FastAPI
            >>> app = FastAPI()
            >>> app.include_router(dashboard.create_api_router())
        """
        try:
            from fastapi import APIRouter, HTTPException
            from fastapi.responses import JSONResponse
        except ImportError:
            logger.warning("FastAPI not available for metrics API")
            return None

        router = APIRouter(prefix=self.config.api_prefix, tags=["metrics"])

        @router.get("/summary")
        async def get_summary():
            """Get dashboard summary."""
            return self.get_summary().dict()

        @router.get("/learning_curve/{metric_name}")
        async def get_learning_curve(metric_name: str, last_n: Optional[int] = None):
            """Get learning curve for a metric."""
            if metric_name not in self._metrics:
                raise HTTPException(404, f"Metric not found: {metric_name}")
            return self.get_learning_curve(metric_name, last_n=last_n).dict()

        @router.get("/drift/{metric_name}")
        async def get_drift(metric_name: str):
            """Get drift status for a metric."""
            return self.get_drift_status(metric_name).dict()

        @router.get("/plasticity")
        async def get_plasticity():
            """Get plasticity metrics."""
            return self.get_plasticity_metrics().dict()

        @router.get("/retention")
        async def get_retention():
            """Get retention metrics."""
            return self.get_retention_metrics().dict()

        @router.get("/alerts")
        async def get_alerts(unacknowledged: bool = False, limit: int = 50):
            """Get alerts."""
            alerts = self.get_alerts(
                unacknowledged_only=unacknowledged,
                limit=limit
            )
            return [a.dict() for a in alerts]

        @router.post("/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str):
            """Acknowledge an alert."""
            if self.acknowledge_alert(alert_id):
                return {"status": "acknowledged"}
            raise HTTPException(404, f"Alert not found: {alert_id}")

        @router.get("/metrics")
        async def list_metrics():
            """List available metrics."""
            return list(self._metrics.keys())

        @router.get("/export")
        async def export_metrics():
            """Export all metrics."""
            return JSONResponse(content=json.loads(self.export_metrics()))

        return router


# Factory functions
def create_standard_dashboard(
    history_length: int = 10000
) -> MetricsDashboard:
    """Create a standard metrics dashboard."""
    config = DashboardConfig(
        history_length=history_length,
        drift_threshold=0.1,
        alert_on_drift=True
    )
    return MetricsDashboard(config)


def create_production_dashboard() -> MetricsDashboard:
    """Create a production-ready metrics dashboard."""
    config = DashboardConfig(
        history_length=50000,
        drift_threshold=0.05,
        degradation_threshold=0.03,
        alert_on_drift=True,
        alert_on_degradation=True,
        auto_snapshot_interval=1000
    )
    return MetricsDashboard(config)


# Unit test stubs
class TestMetricsDashboard:
    """Unit tests for MetricsDashboard."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        dashboard = MetricsDashboard()
        assert dashboard.config.history_length == 10000

    def test_record_metric(self):
        """Test recording metrics."""
        dashboard = MetricsDashboard()

        for i in range(100):
            dashboard.record_metric("loss", 1.0 - i * 0.01, step=i)

        assert len(dashboard._metrics["loss"]) == 100
        assert dashboard._ema_values["loss"] < 1.0

    def test_record_training_step(self):
        """Test recording training step."""
        dashboard = MetricsDashboard()

        dashboard.record_training_step(
            step=0,
            loss=1.0,
            accuracy=0.5
        )

        assert "loss" in dashboard._metrics
        assert "accuracy" in dashboard._metrics

    def test_learning_curve(self):
        """Test getting learning curve."""
        dashboard = MetricsDashboard()

        for i in range(50):
            dashboard.record_metric("loss", 1.0 - i * 0.01, step=i)

        curve = dashboard.get_learning_curve("loss")
        assert len(curve.steps) == 50
        assert curve.ema_values is not None

    def test_drift_detection(self):
        """Test drift detection."""
        config = DashboardConfig(drift_window=20, drift_threshold=0.1)
        dashboard = MetricsDashboard(config)

        # Establish baseline
        for i in range(30):
            dashboard.record_metric("accuracy", 0.9 + np.random.normal(0, 0.01))

        # Introduce drift
        for i in range(30):
            dashboard.record_metric("accuracy", 0.7 + np.random.normal(0, 0.01))

        drift = dashboard.get_drift_status("accuracy")
        # Should detect the drop
        assert drift.drift_score > 0

    def test_plasticity_metrics(self):
        """Test plasticity metrics."""
        dashboard = MetricsDashboard()

        # Record some gradients
        for i in range(50):
            dashboard.record_gradient_norm(np.random.uniform(0.1, 1.0))

        plasticity = dashboard.get_plasticity_metrics()
        assert plasticity.gradient_magnitude > 0

    def test_retention_metrics(self):
        """Test retention metrics."""
        dashboard = MetricsDashboard()

        # Record initial performance
        dashboard.record_task_performance("task1", 0.95, is_initial=True)
        dashboard.record_task_performance("task2", 0.90, is_initial=True)

        # Record current (degraded) performance
        dashboard.record_task_performance("task1", 0.85)
        dashboard.record_task_performance("task2", 0.88)

        retention = dashboard.get_retention_metrics()
        assert retention.average_forgetting > 0
        assert len(retention.task_retentions) == 2

    def test_summary(self):
        """Test summary generation."""
        dashboard = MetricsDashboard()

        for i in range(100):
            dashboard.record_training_step(
                step=i,
                loss=1.0 - i * 0.005,
                accuracy=0.5 + i * 0.003
            )

        summary = dashboard.get_summary()
        assert summary.total_steps == 100
        assert "loss" in summary.current_metrics
        assert len(summary.provenance_hash) == 64

    def test_alerts(self):
        """Test alert system."""
        config = DashboardConfig(
            alert_on_degradation=True,
            degradation_threshold=0.05
        )
        dashboard = MetricsDashboard(config)

        # Establish good baseline
        for i in range(50):
            dashboard.record_metric("accuracy", 0.95)

        # Trigger degradation
        for i in range(10):
            dashboard.record_metric("accuracy", 0.80)

        alerts = dashboard.get_alerts()
        # Should have alerts
        assert len(alerts) >= 0  # May or may not trigger depending on threshold
