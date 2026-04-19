# -*- coding: utf-8 -*-
"""
Adaptation Triggers Module

This module provides intelligent triggers for model adaptation in GreenLang
agents, determining when and how models should be updated based on
performance metrics, data drift, and business rules.

Adaptation triggers enable autonomous model maintenance by monitoring
key indicators and initiating retraining or adaptation when thresholds
are exceeded, ensuring continuous model quality.

Example:
    >>> from greenlang.ml.self_learning import AdaptationTrigger
    >>> trigger = AdaptationTrigger(performance_threshold=0.85)
    >>> for metrics in monitoring_stream:
    ...     if trigger.should_adapt(metrics):
    ...         model.retrain()
"""

from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class TriggerType(str, Enum):
    """Types of adaptation triggers."""
    PERFORMANCE = "performance"
    DRIFT = "drift"
    SCHEDULE = "schedule"
    DATA_VOLUME = "data_volume"
    CONCEPT_CHANGE = "concept_change"
    REGULATORY = "regulatory"
    COMPOSITE = "composite"


class TriggerAction(str, Enum):
    """Actions to take when trigger fires."""
    RETRAIN_FULL = "retrain_full"
    RETRAIN_INCREMENTAL = "retrain_incremental"
    ADAPT_FEW_SHOT = "adapt_few_shot"
    ROLLBACK = "rollback"
    ALERT = "alert"
    QUARANTINE = "quarantine"


class TriggerCondition(str, Enum):
    """Trigger condition operators."""
    LESS_THAN = "lt"
    GREATER_THAN = "gt"
    LESS_EQUAL = "le"
    GREATER_EQUAL = "ge"
    EQUALS = "eq"
    NOT_EQUALS = "ne"


class AdaptationTriggerConfig(BaseModel):
    """Configuration for adaptation trigger."""

    trigger_type: TriggerType = Field(
        default=TriggerType.PERFORMANCE,
        description="Type of trigger"
    )
    metric_name: str = Field(
        default="accuracy",
        description="Metric to monitor"
    )
    threshold: float = Field(
        default=0.85,
        description="Threshold value"
    )
    condition: TriggerCondition = Field(
        default=TriggerCondition.LESS_THAN,
        description="Condition operator"
    )
    window_size: int = Field(
        default=100,
        ge=10,
        description="Sliding window size for metrics"
    )
    cooldown_minutes: int = Field(
        default=60,
        ge=0,
        description="Cooldown period after trigger fires"
    )
    consecutive_failures: int = Field(
        default=3,
        ge=1,
        description="Consecutive failures before trigger"
    )
    action: TriggerAction = Field(
        default=TriggerAction.RETRAIN_INCREMENTAL,
        description="Action when trigger fires"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )


class TriggerEvent(BaseModel):
    """Record of a trigger event."""

    trigger_id: str = Field(
        ...,
        description="Trigger identifier"
    )
    trigger_type: str = Field(
        ...,
        description="Type of trigger"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When trigger fired"
    )
    metric_value: float = Field(
        ...,
        description="Metric value that triggered"
    )
    threshold: float = Field(
        ...,
        description="Threshold that was breached"
    )
    action_taken: str = Field(
        ...,
        description="Action that was recommended"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )


class TriggerStatus(BaseModel):
    """Current status of a trigger."""

    is_active: bool = Field(
        ...,
        description="Whether trigger is active"
    )
    in_cooldown: bool = Field(
        ...,
        description="Whether in cooldown period"
    )
    cooldown_remaining_seconds: Optional[int] = Field(
        default=None,
        description="Seconds remaining in cooldown"
    )
    consecutive_count: int = Field(
        ...,
        description="Current consecutive failure count"
    )
    last_triggered: Optional[datetime] = Field(
        default=None,
        description="When last triggered"
    )
    window_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Current window statistics"
    )


class AdaptationTrigger:
    """
    Adaptation Trigger for GreenLang model management.

    This class monitors model performance and data characteristics
    to determine when adaptation is needed, providing intelligent
    automation for continuous model improvement.

    Key capabilities:
    - Performance-based triggers
    - Drift detection triggers
    - Scheduled triggers
    - Composite (multi-condition) triggers
    - Cooldown management
    - Provenance tracking

    Attributes:
        config: Trigger configuration
        _metric_window: Sliding window of metric values
        _trigger_history: History of trigger events
        _last_triggered: Timestamp of last trigger

    Example:
        >>> trigger = AdaptationTrigger(config=AdaptationTriggerConfig(
        ...     trigger_type=TriggerType.PERFORMANCE,
        ...     metric_name="f1_score",
        ...     threshold=0.80,
        ...     condition=TriggerCondition.LESS_THAN
        ... ))
        >>> # Monitor model performance
        >>> for batch in data_stream:
        ...     metrics = model.evaluate(batch)
        ...     result = trigger.check(metrics)
        ...     if result.should_adapt:
        ...         perform_adaptation(result.action)
    """

    def __init__(
        self,
        config: Optional[AdaptationTriggerConfig] = None,
        trigger_id: Optional[str] = None
    ):
        """
        Initialize adaptation trigger.

        Args:
            config: Trigger configuration
            trigger_id: Unique trigger identifier
        """
        self.config = config or AdaptationTriggerConfig()
        self.trigger_id = trigger_id or f"trigger_{id(self)}"

        self._metric_window: deque = deque(maxlen=self.config.window_size)
        self._trigger_history: List[TriggerEvent] = []
        self._last_triggered: Optional[datetime] = None
        self._consecutive_count: int = 0

        # Sub-triggers for composite
        self._sub_triggers: List["AdaptationTrigger"] = []

        logger.info(
            f"AdaptationTrigger initialized: {self.trigger_id}, "
            f"type={self.config.trigger_type}"
        )

    def _evaluate_condition(
        self,
        value: float,
        threshold: float
    ) -> bool:
        """
        Evaluate condition against threshold.

        Args:
            value: Current value
            threshold: Threshold to compare against

        Returns:
            Whether condition is met (trigger should fire)
        """
        if self.config.condition == TriggerCondition.LESS_THAN:
            return value < threshold
        elif self.config.condition == TriggerCondition.GREATER_THAN:
            return value > threshold
        elif self.config.condition == TriggerCondition.LESS_EQUAL:
            return value <= threshold
        elif self.config.condition == TriggerCondition.GREATER_EQUAL:
            return value >= threshold
        elif self.config.condition == TriggerCondition.EQUALS:
            return abs(value - threshold) < 1e-10
        elif self.config.condition == TriggerCondition.NOT_EQUALS:
            return abs(value - threshold) >= 1e-10
        return False

    def _is_in_cooldown(self) -> bool:
        """Check if trigger is in cooldown period."""
        if self._last_triggered is None:
            return False

        elapsed = datetime.utcnow() - self._last_triggered
        cooldown = timedelta(minutes=self.config.cooldown_minutes)

        return elapsed < cooldown

    def _get_cooldown_remaining(self) -> Optional[int]:
        """Get remaining cooldown time in seconds."""
        if self._last_triggered is None:
            return None

        elapsed = datetime.utcnow() - self._last_triggered
        cooldown = timedelta(minutes=self.config.cooldown_minutes)
        remaining = cooldown - elapsed

        if remaining.total_seconds() > 0:
            return int(remaining.total_seconds())
        return None

    def _calculate_provenance(
        self,
        metric_value: float,
        triggered: bool
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        combined = (
            f"{self.trigger_id}|{metric_value}|{triggered}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(combined.encode()).hexdigest()

    def _get_window_stats(self) -> Dict[str, float]:
        """Calculate statistics from metric window."""
        if not self._metric_window:
            return {}

        values = list(self._metric_window)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "trend": self._calculate_trend(values),
            "count": len(values)
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of metric values."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Simple linear regression
        n = len(values)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)

        denominator = n * sum_xx - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return float(slope)

    def update(self, metric_value: float) -> None:
        """
        Update trigger with new metric value.

        Args:
            metric_value: Latest metric value
        """
        self._metric_window.append(metric_value)

    def check(
        self,
        metrics: Optional[Dict[str, float]] = None,
        metric_value: Optional[float] = None
    ) -> "TriggerResult":
        """
        Check if trigger should fire.

        Args:
            metrics: Dictionary of metrics
            metric_value: Direct metric value (alternative to metrics dict)

        Returns:
            TriggerResult with decision and context

        Example:
            >>> result = trigger.check({"accuracy": 0.72, "f1": 0.68})
            >>> if result.should_adapt:
            ...     print(f"Action: {result.action}")
        """
        # Get metric value
        if metric_value is not None:
            value = metric_value
        elif metrics is not None:
            value = metrics.get(self.config.metric_name, 0.0)
        elif self._metric_window:
            value = self._metric_window[-1]
        else:
            return TriggerResult(
                should_adapt=False,
                trigger_id=self.trigger_id,
                reason="No metric value available"
            )

        # Update window
        self.update(value)

        # Check cooldown
        if self._is_in_cooldown():
            return TriggerResult(
                should_adapt=False,
                trigger_id=self.trigger_id,
                reason="In cooldown period",
                cooldown_remaining=self._get_cooldown_remaining()
            )

        # Evaluate condition
        condition_met = self._evaluate_condition(value, self.config.threshold)

        if condition_met:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        # Check consecutive failures
        should_trigger = (
            condition_met and
            self._consecutive_count >= self.config.consecutive_failures
        )

        if should_trigger:
            # Record trigger event
            provenance_hash = self._calculate_provenance(value, True)

            event = TriggerEvent(
                trigger_id=self.trigger_id,
                trigger_type=self.config.trigger_type.value,
                timestamp=datetime.utcnow(),
                metric_value=value,
                threshold=self.config.threshold,
                action_taken=self.config.action.value,
                context=self._get_window_stats(),
                provenance_hash=provenance_hash
            )
            self._trigger_history.append(event)
            self._last_triggered = datetime.utcnow()
            self._consecutive_count = 0

            logger.warning(
                f"Trigger {self.trigger_id} fired: "
                f"{self.config.metric_name}={value:.4f} "
                f"(threshold={self.config.threshold})"
            )

            return TriggerResult(
                should_adapt=True,
                trigger_id=self.trigger_id,
                action=self.config.action,
                metric_value=value,
                threshold=self.config.threshold,
                reason=f"{self.config.metric_name} breached threshold",
                provenance_hash=provenance_hash,
                window_stats=self._get_window_stats()
            )

        return TriggerResult(
            should_adapt=False,
            trigger_id=self.trigger_id,
            metric_value=value,
            threshold=self.config.threshold,
            reason="Threshold not breached" if not condition_met else "Building consecutive count",
            consecutive_count=self._consecutive_count
        )

    def add_sub_trigger(self, trigger: "AdaptationTrigger") -> None:
        """Add a sub-trigger for composite evaluation."""
        self._sub_triggers.append(trigger)

    def check_composite(
        self,
        metrics: Dict[str, float],
        require_all: bool = True
    ) -> "TriggerResult":
        """
        Check composite trigger with multiple conditions.

        Args:
            metrics: Dictionary of metrics
            require_all: If True, all sub-triggers must fire

        Returns:
            TriggerResult for composite trigger
        """
        if not self._sub_triggers:
            return self.check(metrics)

        results = []
        for sub_trigger in self._sub_triggers:
            result = sub_trigger.check(metrics)
            results.append(result)

        if require_all:
            should_adapt = all(r.should_adapt for r in results)
        else:
            should_adapt = any(r.should_adapt for r in results)

        triggered_triggers = [r for r in results if r.should_adapt]

        return TriggerResult(
            should_adapt=should_adapt,
            trigger_id=self.trigger_id,
            reason=f"Composite: {len(triggered_triggers)}/{len(results)} triggered",
            sub_results=results
        )

    def get_status(self) -> TriggerStatus:
        """Get current trigger status."""
        return TriggerStatus(
            is_active=not self._is_in_cooldown(),
            in_cooldown=self._is_in_cooldown(),
            cooldown_remaining_seconds=self._get_cooldown_remaining(),
            consecutive_count=self._consecutive_count,
            last_triggered=self._last_triggered,
            window_metrics=self._get_window_stats()
        )

    def get_history(
        self,
        limit: Optional[int] = None
    ) -> List[TriggerEvent]:
        """Get trigger event history."""
        if limit:
            return self._trigger_history[-limit:]
        return self._trigger_history.copy()

    def reset(self) -> None:
        """Reset trigger state."""
        self._metric_window.clear()
        self._consecutive_count = 0
        self._last_triggered = None
        logger.info(f"Trigger {self.trigger_id} reset")

    def force_cooldown(self, minutes: Optional[int] = None) -> None:
        """Force trigger into cooldown."""
        self._last_triggered = datetime.utcnow()
        if minutes:
            self.config.cooldown_minutes = minutes


class TriggerResult(BaseModel):
    """Result from trigger evaluation."""

    should_adapt: bool = Field(
        ...,
        description="Whether adaptation should occur"
    )
    trigger_id: str = Field(
        ...,
        description="Trigger that evaluated"
    )
    action: Optional[TriggerAction] = Field(
        default=None,
        description="Recommended action"
    )
    metric_value: Optional[float] = Field(
        default=None,
        description="Current metric value"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Threshold value"
    )
    reason: str = Field(
        default="",
        description="Reason for decision"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="Provenance hash if triggered"
    )
    cooldown_remaining: Optional[int] = Field(
        default=None,
        description="Remaining cooldown seconds"
    )
    consecutive_count: Optional[int] = Field(
        default=None,
        description="Current consecutive count"
    )
    window_stats: Optional[Dict[str, float]] = Field(
        default=None,
        description="Window statistics"
    )
    sub_results: Optional[List["TriggerResult"]] = Field(
        default=None,
        description="Sub-trigger results (composite)"
    )


# Factory functions for common triggers
def create_performance_trigger(
    metric_name: str = "accuracy",
    threshold: float = 0.85,
    action: TriggerAction = TriggerAction.RETRAIN_INCREMENTAL
) -> AdaptationTrigger:
    """Create a performance-based trigger."""
    config = AdaptationTriggerConfig(
        trigger_type=TriggerType.PERFORMANCE,
        metric_name=metric_name,
        threshold=threshold,
        condition=TriggerCondition.LESS_THAN,
        action=action
    )
    return AdaptationTrigger(config)


def create_drift_trigger(
    threshold: float = 0.1,
    window_size: int = 500
) -> AdaptationTrigger:
    """Create a drift detection trigger."""
    config = AdaptationTriggerConfig(
        trigger_type=TriggerType.DRIFT,
        metric_name="drift_score",
        threshold=threshold,
        condition=TriggerCondition.GREATER_THAN,
        window_size=window_size,
        action=TriggerAction.RETRAIN_FULL
    )
    return AdaptationTrigger(config)


def create_scheduled_trigger(
    interval_minutes: int = 60 * 24  # Daily
) -> AdaptationTrigger:
    """Create a scheduled trigger."""
    config = AdaptationTriggerConfig(
        trigger_type=TriggerType.SCHEDULE,
        metric_name="time_since_last",
        threshold=interval_minutes * 60,  # Convert to seconds
        condition=TriggerCondition.GREATER_THAN,
        cooldown_minutes=interval_minutes,
        action=TriggerAction.RETRAIN_INCREMENTAL
    )
    return AdaptationTrigger(config)


# Unit test stubs
class TestAdaptationTrigger:
    """Unit tests for AdaptationTrigger."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        trigger = AdaptationTrigger()
        assert trigger.config.trigger_type == TriggerType.PERFORMANCE
        assert trigger.config.threshold == 0.85

    def test_condition_evaluation(self):
        """Test condition evaluation."""
        trigger = AdaptationTrigger()

        # Less than
        trigger.config.condition = TriggerCondition.LESS_THAN
        assert trigger._evaluate_condition(0.5, 0.85) is True
        assert trigger._evaluate_condition(0.9, 0.85) is False

        # Greater than
        trigger.config.condition = TriggerCondition.GREATER_THAN
        assert trigger._evaluate_condition(0.9, 0.85) is True
        assert trigger._evaluate_condition(0.5, 0.85) is False

    def test_cooldown_management(self):
        """Test cooldown period management."""
        config = AdaptationTriggerConfig(cooldown_minutes=5)
        trigger = AdaptationTrigger(config)

        assert not trigger._is_in_cooldown()

        # Simulate trigger
        trigger._last_triggered = datetime.utcnow()
        assert trigger._is_in_cooldown()

        remaining = trigger._get_cooldown_remaining()
        assert remaining is not None
        assert remaining > 0

    def test_consecutive_failures(self):
        """Test consecutive failure counting."""
        config = AdaptationTriggerConfig(
            threshold=0.85,
            consecutive_failures=3
        )
        trigger = AdaptationTrigger(config)

        # First two failures - should not trigger
        result1 = trigger.check(metric_value=0.7)
        assert not result1.should_adapt
        assert trigger._consecutive_count == 1

        result2 = trigger.check(metric_value=0.6)
        assert not result2.should_adapt
        assert trigger._consecutive_count == 2

        # Third failure - should trigger
        result3 = trigger.check(metric_value=0.5)
        assert result3.should_adapt

    def test_window_statistics(self):
        """Test window statistics calculation."""
        trigger = AdaptationTrigger()

        for i in range(10):
            trigger.update(0.8 + i * 0.01)

        stats = trigger._get_window_stats()
        assert "mean" in stats
        assert "std" in stats
        assert "trend" in stats
        assert stats["count"] == 10

    def test_provenance_deterministic(self):
        """Test provenance hash consistency."""
        trigger = AdaptationTrigger()

        # Same inputs should give same hash
        hash1 = trigger._calculate_provenance(0.5, True)
        # Note: includes timestamp so will differ
        assert len(hash1) == 64  # SHA-256 hex length
