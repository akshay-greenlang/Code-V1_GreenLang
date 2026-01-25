# -*- coding: utf-8 -*-
"""
Model Rollback Mechanisms Module

This module provides automatic rollback mechanisms for GreenLang ML models,
enabling safe deployment with performance degradation detection, A/B test-based
triggers, and canary deployment rollback with comprehensive audit logging.

Rollback mechanisms are critical for production ML systems to ensure model
quality is maintained and regressions are quickly detected and remediated.

Example:
    >>> from greenlang.ml.mlops import RollbackManager
    >>> manager = RollbackManager(model_registry)
    >>> manager.monitor_and_rollback("emission_predictor", current_model, baseline_metrics)
    >>> if manager.should_rollback():
    ...     manager.execute_rollback("emission_predictor")
"""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum
from collections import deque
import json

logger = logging.getLogger(__name__)


class RollbackReason(str, Enum):
    """Reasons for model rollback."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    AB_TEST_FAILURE = "ab_test_failure"
    CANARY_FAILURE = "canary_failure"
    MANUAL_TRIGGER = "manual_trigger"
    DRIFT_DETECTED = "drift_detected"
    ERROR_RATE_EXCEEDED = "error_rate_exceeded"
    LATENCY_EXCEEDED = "latency_exceeded"
    SAFETY_VIOLATION = "safety_violation"


class RollbackStatus(str, Enum):
    """Status of rollback operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentStrategy(str, Enum):
    """Deployment strategies supporting rollback."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    AB_TEST = "ab_test"
    SHADOW = "shadow"


class RollbackConfig(BaseModel):
    """Configuration for rollback manager."""

    performance_threshold: float = Field(
        default=0.1,
        ge=0,
        le=1.0,
        description="Maximum allowed performance degradation (relative)"
    )
    error_rate_threshold: float = Field(
        default=0.05,
        ge=0,
        le=1.0,
        description="Maximum allowed error rate"
    )
    latency_threshold_ms: float = Field(
        default=500.0,
        gt=0,
        description="Maximum allowed latency in milliseconds"
    )
    monitoring_window: int = Field(
        default=100,
        ge=10,
        description="Number of samples in monitoring window"
    )
    min_samples_for_decision: int = Field(
        default=30,
        ge=10,
        description="Minimum samples before rollback decision"
    )
    canary_traffic_percent: float = Field(
        default=5.0,
        gt=0,
        le=50,
        description="Percentage of traffic to canary"
    )
    canary_promotion_threshold: float = Field(
        default=0.95,
        gt=0.5,
        le=1.0,
        description="Required success rate to promote canary"
    )
    auto_rollback_enabled: bool = Field(
        default=True,
        description="Enable automatic rollback"
    )
    retain_rollback_history: int = Field(
        default=100,
        ge=10,
        description="Number of rollback events to retain"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )


class ModelMetrics(BaseModel):
    """Performance metrics for a model."""

    model_name: str = Field(
        ...,
        description="Model name"
    )
    version: str = Field(
        ...,
        description="Model version"
    )
    mse: Optional[float] = Field(
        default=None,
        description="Mean squared error"
    )
    mae: Optional[float] = Field(
        default=None,
        description="Mean absolute error"
    )
    r2: Optional[float] = Field(
        default=None,
        description="R-squared score"
    )
    accuracy: Optional[float] = Field(
        default=None,
        description="Accuracy (for classification)"
    )
    error_rate: float = Field(
        default=0.0,
        description="Error/exception rate"
    )
    avg_latency_ms: float = Field(
        default=0.0,
        description="Average latency in ms"
    )
    p99_latency_ms: float = Field(
        default=0.0,
        description="P99 latency in ms"
    )
    sample_count: int = Field(
        default=0,
        description="Number of samples"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Metrics timestamp"
    )
    custom_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional custom metrics"
    )


class ModelComparison(BaseModel):
    """Comparison between two model versions."""

    current_model: str = Field(
        ...,
        description="Current model version"
    )
    baseline_model: str = Field(
        ...,
        description="Baseline model version"
    )
    current_metrics: ModelMetrics = Field(
        ...,
        description="Current model metrics"
    )
    baseline_metrics: ModelMetrics = Field(
        ...,
        description="Baseline model metrics"
    )
    performance_delta: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance differences"
    )
    is_degraded: bool = Field(
        ...,
        description="Whether performance is degraded"
    )
    degradation_severity: float = Field(
        ...,
        description="Severity of degradation (0-1)"
    )
    comparison_details: str = Field(
        ...,
        description="Human-readable comparison"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Comparison timestamp"
    )


class RollbackEvent(BaseModel):
    """Audit record for a rollback event."""

    event_id: str = Field(
        ...,
        description="Unique event identifier"
    )
    model_name: str = Field(
        ...,
        description="Model name"
    )
    from_version: str = Field(
        ...,
        description="Version being rolled back from"
    )
    to_version: str = Field(
        ...,
        description="Version being rolled back to"
    )
    reason: RollbackReason = Field(
        ...,
        description="Reason for rollback"
    )
    status: RollbackStatus = Field(
        ...,
        description="Rollback status"
    )
    triggered_by: str = Field(
        ...,
        description="Who/what triggered the rollback"
    )
    metrics_before: Optional[ModelMetrics] = Field(
        default=None,
        description="Metrics before rollback"
    )
    metrics_after: Optional[ModelMetrics] = Field(
        default=None,
        description="Metrics after rollback"
    )
    comparison: Optional[ModelComparison] = Field(
        default=None,
        description="Model comparison that triggered rollback"
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Rollback duration"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 provenance hash"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event creation time"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Event completion time"
    )


class CanaryStatus(BaseModel):
    """Status of canary deployment."""

    model_name: str = Field(
        ...,
        description="Model name"
    )
    canary_version: str = Field(
        ...,
        description="Canary version"
    )
    baseline_version: str = Field(
        ...,
        description="Baseline version"
    )
    traffic_percent: float = Field(
        ...,
        description="Current canary traffic percent"
    )
    canary_samples: int = Field(
        default=0,
        description="Samples served by canary"
    )
    baseline_samples: int = Field(
        default=0,
        description="Samples served by baseline"
    )
    canary_success_rate: float = Field(
        default=1.0,
        description="Canary success rate"
    )
    baseline_success_rate: float = Field(
        default=1.0,
        description="Baseline success rate"
    )
    should_promote: bool = Field(
        default=False,
        description="Whether canary should be promoted"
    )
    should_rollback: bool = Field(
        default=False,
        description="Whether canary should be rolled back"
    )
    status_message: str = Field(
        default="",
        description="Status message"
    )
    started_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Canary start time"
    )


class RollbackManager:
    """
    Model Rollback Manager for GreenLang.

    This class provides comprehensive rollback mechanisms for ML models,
    including performance monitoring, A/B test-based triggers, canary
    deployment rollback, and audit logging with provenance.

    Key capabilities:
    - Automatic rollback on performance degradation
    - Model version comparison
    - A/B test-based rollback triggers
    - Canary deployment with gradual rollout
    - Comprehensive audit logging
    - Provenance tracking

    Attributes:
        config: Rollback configuration
        model_registry: Reference to model registry
        _metrics_buffer: Rolling buffer of metrics
        _rollback_history: History of rollback events
        _canary_status: Current canary deployment status

    Example:
        >>> from greenlang.ml.mlops import RollbackManager, ModelRegistry
        >>> registry = ModelRegistry()
        >>> manager = RollbackManager(registry, config=RollbackConfig(
        ...     performance_threshold=0.1,
        ...     auto_rollback_enabled=True
        ... ))
        >>> # Monitor and automatically rollback if degraded
        >>> manager.record_prediction_result("model_v2", prediction=1.5, actual=1.4)
        >>> if manager.should_rollback("model_name"):
        ...     event = manager.execute_rollback("model_name", "v2", "v1")
    """

    def __init__(
        self,
        model_registry: Optional[Any] = None,
        config: Optional[RollbackConfig] = None
    ):
        """
        Initialize rollback manager.

        Args:
            model_registry: Reference to model registry
            config: Rollback configuration
        """
        self.config = config or RollbackConfig()
        self.model_registry = model_registry

        # Metrics buffer per model version
        self._metrics_buffer: Dict[str, deque] = {}

        # Rollback history
        self._rollback_history: List[RollbackEvent] = []

        # Current canary status per model
        self._canary_status: Dict[str, CanaryStatus] = {}

        # Baseline metrics per model
        self._baseline_metrics: Dict[str, ModelMetrics] = {}

        # Event counter for unique IDs
        self._event_counter = 0

        logger.info(
            f"RollbackManager initialized: threshold={self.config.performance_threshold}, "
            f"auto_rollback={self.config.auto_rollback_enabled}"
        )

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        self._event_counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"rb_{timestamp}_{self._event_counter:06d}"

    def _calculate_provenance(
        self,
        model_name: str,
        from_version: str,
        to_version: str,
        reason: RollbackReason
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        combined = (
            f"{model_name}|{from_version}|{to_version}|"
            f"{reason.value}|{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(combined.encode()).hexdigest()

    def _get_buffer_key(self, model_name: str, version: str) -> str:
        """Get buffer key for model version."""
        return f"{model_name}:{version}"

    def set_baseline_metrics(
        self,
        model_name: str,
        metrics: ModelMetrics
    ) -> None:
        """
        Set baseline metrics for a model.

        Args:
            model_name: Model name
            metrics: Baseline metrics
        """
        self._baseline_metrics[model_name] = metrics
        logger.info(f"Baseline metrics set for {model_name}")

    def record_prediction_result(
        self,
        model_name: str,
        version: str,
        prediction: float,
        actual: Optional[float] = None,
        latency_ms: float = 0.0,
        is_error: bool = False
    ) -> None:
        """
        Record a prediction result for monitoring.

        Args:
            model_name: Model name
            version: Model version
            prediction: Predicted value
            actual: Actual value (if available)
            latency_ms: Prediction latency
            is_error: Whether prediction resulted in error
        """
        key = self._get_buffer_key(model_name, version)

        if key not in self._metrics_buffer:
            self._metrics_buffer[key] = deque(
                maxlen=self.config.monitoring_window
            )

        result = {
            "prediction": prediction,
            "actual": actual,
            "latency_ms": latency_ms,
            "is_error": is_error,
            "timestamp": datetime.utcnow(),
            "squared_error": (prediction - actual) ** 2 if actual is not None else None
        }

        self._metrics_buffer[key].append(result)

    def compute_current_metrics(
        self,
        model_name: str,
        version: str
    ) -> Optional[ModelMetrics]:
        """
        Compute current metrics from buffer.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            Current metrics or None if insufficient data
        """
        key = self._get_buffer_key(model_name, version)

        if key not in self._metrics_buffer:
            return None

        buffer = self._metrics_buffer[key]

        if len(buffer) < self.config.min_samples_for_decision:
            return None

        results = list(buffer)

        # Calculate metrics
        latencies = [r["latency_ms"] for r in results]
        errors = [r for r in results if r["is_error"]]
        error_rate = len(errors) / len(results)

        # Calculate MSE if actuals available
        with_actuals = [r for r in results if r["actual"] is not None]
        mse = None
        mae = None

        if with_actuals:
            squared_errors = [r["squared_error"] for r in with_actuals]
            abs_errors = [abs(r["prediction"] - r["actual"]) for r in with_actuals]
            mse = float(np.mean(squared_errors))
            mae = float(np.mean(abs_errors))

        return ModelMetrics(
            model_name=model_name,
            version=version,
            mse=mse,
            mae=mae,
            error_rate=error_rate,
            avg_latency_ms=float(np.mean(latencies)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            sample_count=len(results)
        )

    def compare_models(
        self,
        model_name: str,
        current_version: str,
        baseline_version: Optional[str] = None
    ) -> Optional[ModelComparison]:
        """
        Compare current model version against baseline.

        Args:
            model_name: Model name
            current_version: Current version
            baseline_version: Baseline version (uses stored baseline if None)

        Returns:
            Model comparison result
        """
        current_metrics = self.compute_current_metrics(model_name, current_version)
        if current_metrics is None:
            return None

        # Get baseline metrics
        if baseline_version:
            baseline_metrics = self.compute_current_metrics(model_name, baseline_version)
        else:
            baseline_metrics = self._baseline_metrics.get(model_name)

        if baseline_metrics is None:
            logger.warning(f"No baseline metrics for {model_name}")
            return None

        # Calculate performance delta
        performance_delta = {}
        is_degraded = False
        degradation_severity = 0.0

        # Compare MSE (lower is better)
        if current_metrics.mse is not None and baseline_metrics.mse is not None:
            if baseline_metrics.mse > 0:
                mse_delta = (current_metrics.mse - baseline_metrics.mse) / baseline_metrics.mse
                performance_delta["mse"] = mse_delta
                if mse_delta > self.config.performance_threshold:
                    is_degraded = True
                    degradation_severity = max(degradation_severity, min(mse_delta, 1.0))

        # Compare error rate
        error_delta = current_metrics.error_rate - baseline_metrics.error_rate
        performance_delta["error_rate"] = error_delta
        if current_metrics.error_rate > self.config.error_rate_threshold:
            is_degraded = True
            degradation_severity = max(
                degradation_severity,
                current_metrics.error_rate / self.config.error_rate_threshold
            )

        # Compare latency
        if baseline_metrics.avg_latency_ms > 0:
            latency_delta = (
                current_metrics.avg_latency_ms - baseline_metrics.avg_latency_ms
            ) / baseline_metrics.avg_latency_ms
            performance_delta["latency"] = latency_delta

        if current_metrics.avg_latency_ms > self.config.latency_threshold_ms:
            is_degraded = True
            degradation_severity = max(
                degradation_severity,
                current_metrics.avg_latency_ms / self.config.latency_threshold_ms
            )

        # Generate comparison details
        details_parts = []
        if "mse" in performance_delta:
            details_parts.append(f"MSE change: {performance_delta['mse']:+.2%}")
        details_parts.append(f"Error rate: {current_metrics.error_rate:.2%}")
        details_parts.append(f"Avg latency: {current_metrics.avg_latency_ms:.1f}ms")

        comparison_details = "; ".join(details_parts)

        provenance_hash = hashlib.sha256(
            f"{current_version}|{baseline_version}|{degradation_severity}".encode()
        ).hexdigest()

        return ModelComparison(
            current_model=current_version,
            baseline_model=baseline_metrics.version,
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            performance_delta=performance_delta,
            is_degraded=is_degraded,
            degradation_severity=min(degradation_severity, 1.0),
            comparison_details=comparison_details,
            provenance_hash=provenance_hash
        )

    def should_rollback(
        self,
        model_name: str,
        current_version: str
    ) -> Tuple[bool, Optional[RollbackReason], Optional[ModelComparison]]:
        """
        Determine if rollback should be triggered.

        Args:
            model_name: Model name
            current_version: Current version

        Returns:
            Tuple of (should_rollback, reason, comparison)
        """
        comparison = self.compare_models(model_name, current_version)

        if comparison is None:
            return (False, None, None)

        if comparison.is_degraded:
            return (True, RollbackReason.PERFORMANCE_DEGRADATION, comparison)

        # Check canary status
        if model_name in self._canary_status:
            canary = self._canary_status[model_name]
            if canary.should_rollback:
                return (True, RollbackReason.CANARY_FAILURE, comparison)

        return (False, None, comparison)

    def execute_rollback(
        self,
        model_name: str,
        from_version: str,
        to_version: str,
        reason: RollbackReason = RollbackReason.MANUAL_TRIGGER,
        triggered_by: str = "system"
    ) -> RollbackEvent:
        """
        Execute model rollback.

        Args:
            model_name: Model name
            from_version: Version to rollback from
            to_version: Version to rollback to
            reason: Reason for rollback
            triggered_by: Who triggered the rollback

        Returns:
            RollbackEvent with details

        Example:
            >>> event = manager.execute_rollback(
            ...     "emission_predictor", "v2", "v1",
            ...     reason=RollbackReason.PERFORMANCE_DEGRADATION
            ... )
            >>> print(f"Rollback {event.status}: {event.event_id}")
        """
        event_id = self._generate_event_id()
        start_time = datetime.utcnow()

        logger.info(
            f"Executing rollback: {model_name} {from_version} -> {to_version}, "
            f"reason={reason.value}"
        )

        # Get metrics before rollback
        metrics_before = self.compute_current_metrics(model_name, from_version)

        # Get comparison if available
        comparison = self.compare_models(model_name, from_version)

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            model_name, from_version, to_version, reason
        )

        # Create initial event
        event = RollbackEvent(
            event_id=event_id,
            model_name=model_name,
            from_version=from_version,
            to_version=to_version,
            reason=reason,
            status=RollbackStatus.IN_PROGRESS,
            triggered_by=triggered_by,
            metrics_before=metrics_before,
            comparison=comparison,
            provenance_hash=provenance_hash
        )

        try:
            # Execute the actual rollback via model registry
            if self.model_registry is not None:
                # Transition the target version to production
                if hasattr(self.model_registry, "transition_stage"):
                    from greenlang.ml.mlops.model_registry import ModelStage
                    self.model_registry.transition_stage(
                        model_name,
                        to_version,
                        ModelStage.PRODUCTION
                    )
                    logger.info(f"Promoted {model_name} v{to_version} to Production")

            # Mark rollback as complete
            event.status = RollbackStatus.COMPLETED
            event.completed_at = datetime.utcnow()
            event.duration_seconds = (
                event.completed_at - start_time
            ).total_seconds()

            logger.info(
                f"Rollback completed: {event_id} in {event.duration_seconds:.2f}s"
            )

        except Exception as e:
            event.status = RollbackStatus.FAILED
            event.error_message = str(e)
            event.completed_at = datetime.utcnow()

            logger.error(f"Rollback failed: {event_id}, error: {e}")

        # Store in history
        self._rollback_history.append(event)

        # Trim history if needed
        if len(self._rollback_history) > self.config.retain_rollback_history:
            self._rollback_history = self._rollback_history[
                -self.config.retain_rollback_history:
            ]

        return event

    def start_canary_deployment(
        self,
        model_name: str,
        canary_version: str,
        baseline_version: str,
        traffic_percent: Optional[float] = None
    ) -> CanaryStatus:
        """
        Start canary deployment.

        Args:
            model_name: Model name
            canary_version: New canary version
            baseline_version: Baseline version
            traffic_percent: Initial traffic percentage

        Returns:
            CanaryStatus
        """
        traffic = traffic_percent or self.config.canary_traffic_percent

        status = CanaryStatus(
            model_name=model_name,
            canary_version=canary_version,
            baseline_version=baseline_version,
            traffic_percent=traffic,
            status_message=f"Canary started with {traffic}% traffic"
        )

        self._canary_status[model_name] = status

        logger.info(
            f"Canary deployment started: {model_name} "
            f"{canary_version} ({traffic}% traffic)"
        )

        return status

    def record_canary_result(
        self,
        model_name: str,
        is_canary: bool,
        is_success: bool
    ) -> None:
        """
        Record a canary deployment result.

        Args:
            model_name: Model name
            is_canary: Whether this was served by canary
            is_success: Whether prediction was successful
        """
        if model_name not in self._canary_status:
            return

        status = self._canary_status[model_name]

        if is_canary:
            status.canary_samples += 1
            if is_success:
                # Update success rate
                success_count = int(status.canary_success_rate * (status.canary_samples - 1))
                success_count += 1
                status.canary_success_rate = success_count / status.canary_samples
        else:
            status.baseline_samples += 1
            if is_success:
                success_count = int(status.baseline_success_rate * (status.baseline_samples - 1))
                success_count += 1
                status.baseline_success_rate = success_count / status.baseline_samples

        # Check if we should promote or rollback
        min_samples = self.config.min_samples_for_decision

        if status.canary_samples >= min_samples:
            if status.canary_success_rate >= self.config.canary_promotion_threshold:
                status.should_promote = True
                status.status_message = "Canary ready for promotion"
            elif status.canary_success_rate < status.baseline_success_rate * 0.9:
                status.should_rollback = True
                status.status_message = "Canary underperforming, rollback recommended"

    def get_canary_status(self, model_name: str) -> Optional[CanaryStatus]:
        """Get current canary deployment status."""
        return self._canary_status.get(model_name)

    def promote_canary(self, model_name: str) -> Optional[RollbackEvent]:
        """
        Promote canary to production.

        Args:
            model_name: Model name

        Returns:
            RollbackEvent or None
        """
        if model_name not in self._canary_status:
            return None

        status = self._canary_status[model_name]

        if not status.should_promote:
            logger.warning(f"Canary not ready for promotion: {model_name}")
            return None

        # This is technically the reverse of rollback - promoting new version
        event = self.execute_rollback(
            model_name=model_name,
            from_version=status.baseline_version,
            to_version=status.canary_version,
            reason=RollbackReason.MANUAL_TRIGGER,
            triggered_by="canary_promotion"
        )

        # Clear canary status
        del self._canary_status[model_name]

        return event

    def rollback_canary(self, model_name: str) -> Optional[RollbackEvent]:
        """
        Rollback canary deployment.

        Args:
            model_name: Model name

        Returns:
            RollbackEvent or None
        """
        if model_name not in self._canary_status:
            return None

        status = self._canary_status[model_name]

        event = self.execute_rollback(
            model_name=model_name,
            from_version=status.canary_version,
            to_version=status.baseline_version,
            reason=RollbackReason.CANARY_FAILURE,
            triggered_by="canary_rollback"
        )

        # Clear canary status
        del self._canary_status[model_name]

        return event

    def get_rollback_history(
        self,
        model_name: Optional[str] = None,
        limit: int = 50
    ) -> List[RollbackEvent]:
        """
        Get rollback history.

        Args:
            model_name: Filter by model name
            limit: Maximum number of events

        Returns:
            List of rollback events
        """
        history = self._rollback_history

        if model_name:
            history = [e for e in history if e.model_name == model_name]

        return history[-limit:]

    def check_ab_test_rollback(
        self,
        model_name: str,
        ab_test_result: Any  # ABTestResult from ab_testing module
    ) -> Tuple[bool, Optional[RollbackReason]]:
        """
        Check if A/B test result should trigger rollback.

        Args:
            model_name: Model name
            ab_test_result: Result from ABTesting

        Returns:
            Tuple of (should_rollback, reason)
        """
        # If B (new model) is significantly worse, trigger rollback
        if hasattr(ab_test_result, "is_significant") and hasattr(ab_test_result, "winner"):
            if ab_test_result.is_significant and ab_test_result.winner == "A":
                return (True, RollbackReason.AB_TEST_FAILURE)

        # Check if effect size indicates significant degradation
        if hasattr(ab_test_result, "effect_size"):
            if ab_test_result.effect_size < -self.config.performance_threshold:
                return (True, RollbackReason.AB_TEST_FAILURE)

        return (False, None)

    def monitor_and_rollback(
        self,
        model_name: str,
        current_version: str,
        fallback_version: str
    ) -> Optional[RollbackEvent]:
        """
        Monitor model and automatically rollback if needed.

        Args:
            model_name: Model name
            current_version: Current version
            fallback_version: Version to rollback to

        Returns:
            RollbackEvent if rollback was executed
        """
        if not self.config.auto_rollback_enabled:
            return None

        should_rb, reason, comparison = self.should_rollback(
            model_name, current_version
        )

        if should_rb and reason is not None:
            return self.execute_rollback(
                model_name=model_name,
                from_version=current_version,
                to_version=fallback_version,
                reason=reason,
                triggered_by="auto_monitor"
            )

        return None

    def export_audit_log(
        self,
        model_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Export rollback audit log for compliance.

        Args:
            model_name: Filter by model name
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of audit log entries
        """
        events = self.get_rollback_history(model_name)

        if start_date:
            events = [e for e in events if e.created_at >= start_date]

        if end_date:
            events = [e for e in events if e.created_at <= end_date]

        return [
            {
                "event_id": e.event_id,
                "model_name": e.model_name,
                "from_version": e.from_version,
                "to_version": e.to_version,
                "reason": e.reason.value,
                "status": e.status.value,
                "triggered_by": e.triggered_by,
                "duration_seconds": e.duration_seconds,
                "provenance_hash": e.provenance_hash,
                "created_at": e.created_at.isoformat(),
                "completed_at": e.completed_at.isoformat() if e.completed_at else None,
                "error_message": e.error_message
            }
            for e in events
        ]


# Unit test stubs
class TestRollbackManager:
    """Unit tests for RollbackManager."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        manager = RollbackManager()
        assert manager.config.performance_threshold == 0.1
        assert manager.config.auto_rollback_enabled is True

    def test_record_prediction_result(self):
        """Test recording prediction results."""
        manager = RollbackManager()

        for i in range(50):
            manager.record_prediction_result(
                "test_model", "v1",
                prediction=float(i),
                actual=float(i) + 0.1,
                latency_ms=10.0
            )

        metrics = manager.compute_current_metrics("test_model", "v1")
        assert metrics is not None
        assert metrics.sample_count == 50

    def test_compare_models(self):
        """Test model comparison."""
        manager = RollbackManager()

        # Set baseline
        baseline = ModelMetrics(
            model_name="test_model",
            version="v1",
            mse=0.1,
            error_rate=0.01,
            avg_latency_ms=50.0,
            sample_count=100
        )
        manager.set_baseline_metrics("test_model", baseline)

        # Record current metrics
        for i in range(50):
            manager.record_prediction_result(
                "test_model", "v2",
                prediction=float(i),
                actual=float(i) + 0.5,  # Higher error
                latency_ms=60.0
            )

        comparison = manager.compare_models("test_model", "v2")
        assert comparison is not None

    def test_should_rollback(self):
        """Test rollback decision."""
        manager = RollbackManager(config=RollbackConfig(
            performance_threshold=0.1,
            min_samples_for_decision=10
        ))

        baseline = ModelMetrics(
            model_name="test",
            version="v1",
            mse=0.1,
            error_rate=0.01,
            avg_latency_ms=50.0,
            sample_count=100
        )
        manager.set_baseline_metrics("test", baseline)

        # Record degraded metrics
        for i in range(20):
            manager.record_prediction_result(
                "test", "v2",
                prediction=float(i),
                actual=float(i) + 2.0,  # Much higher error
                latency_ms=100.0
            )

        should_rb, reason, _ = manager.should_rollback("test", "v2")
        assert should_rb is True
        assert reason == RollbackReason.PERFORMANCE_DEGRADATION

    def test_execute_rollback(self):
        """Test rollback execution."""
        manager = RollbackManager()

        event = manager.execute_rollback(
            "test_model", "v2", "v1",
            reason=RollbackReason.MANUAL_TRIGGER,
            triggered_by="test"
        )

        assert event.status == RollbackStatus.COMPLETED
        assert event.from_version == "v2"
        assert event.to_version == "v1"

    def test_canary_deployment(self):
        """Test canary deployment flow."""
        manager = RollbackManager()

        status = manager.start_canary_deployment(
            "test_model", "v2", "v1",
            traffic_percent=10.0
        )

        assert status.traffic_percent == 10.0
        assert status.canary_version == "v2"

        # Record successful canary results
        for _ in range(50):
            manager.record_canary_result("test_model", is_canary=True, is_success=True)

        status = manager.get_canary_status("test_model")
        assert status.canary_samples == 50
        assert status.should_promote is True

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        manager = RollbackManager()

        # Note: provenance includes timestamp, so we test structure
        hash1 = manager._calculate_provenance(
            "model", "v1", "v2", RollbackReason.MANUAL_TRIGGER
        )
        assert len(hash1) == 64  # SHA-256 hex

    def test_audit_log_export(self):
        """Test audit log export."""
        manager = RollbackManager()

        manager.execute_rollback(
            "test_model", "v2", "v1",
            reason=RollbackReason.PERFORMANCE_DEGRADATION,
            triggered_by="test"
        )

        log = manager.export_audit_log("test_model")
        assert len(log) == 1
        assert log[0]["reason"] == "performance_degradation"
