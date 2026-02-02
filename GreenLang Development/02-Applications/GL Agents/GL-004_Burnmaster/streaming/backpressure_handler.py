"""
Backpressure Handler Module - GL-004 BURNMASTER

This module provides backpressure detection and handling for combustion data
streams, including load shedding strategies and graceful degradation.

Key Features:
    - Queue depth monitoring for backpressure detection
    - Multiple backpressure strategies (rate limiting, sampling, shedding)
    - Priority-based load shedding
    - Automatic recovery to normal operation
    - Comprehensive metrics and monitoring

Example:
    >>> handler = BackpressureHandler(config)
    >>> status = handler.detect_backpressure(queue_depth=1000)
    >>> if status.backpressure_detected:
    ...     result = handler.apply_backpressure("rate_limit")
    ...     shed_result = handler.shed_load("low")

Author: GreenLang Combustion Optimization Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class BackpressureStrategy(str, Enum):
    """Strategies for handling backpressure."""

    RATE_LIMIT = "rate_limit"
    SAMPLING = "sampling"
    PRIORITY_SHED = "priority_shed"
    PAUSE = "pause"
    REJECT = "reject"
    ADAPTIVE = "adaptive"


class BackpressureLevel(str, Enum):
    """Backpressure severity levels."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class LoadPriority(str, Enum):
    """Load priority levels for shedding."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class OperationMode(str, Enum):
    """Handler operation mode."""

    NORMAL = "normal"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    PAUSED = "paused"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class BackpressureConfig(BaseModel):
    """Configuration for backpressure handler."""

    handler_id: str = Field(
        default_factory=lambda: f"bp-{uuid.uuid4().hex[:8]}",
        description="Handler identifier",
    )
    low_threshold: int = Field(
        500,
        ge=10,
        description="Queue depth for low backpressure",
    )
    moderate_threshold: int = Field(
        1000,
        ge=100,
        description="Queue depth for moderate backpressure",
    )
    high_threshold: int = Field(
        2000,
        ge=500,
        description="Queue depth for high backpressure",
    )
    critical_threshold: int = Field(
        5000,
        ge=1000,
        description="Queue depth for critical backpressure",
    )
    default_strategy: BackpressureStrategy = Field(
        BackpressureStrategy.ADAPTIVE,
        description="Default backpressure strategy",
    )
    rate_limit_factor: float = Field(
        0.5,
        ge=0.1,
        le=1.0,
        description="Rate reduction factor for rate limiting",
    )
    sample_rate: float = Field(
        0.5,
        ge=0.01,
        le=1.0,
        description="Sample rate for sampling strategy",
    )
    recovery_threshold: float = Field(
        0.3,
        ge=0.1,
        le=0.9,
        description="Queue fill ratio to trigger recovery",
    )
    recovery_cooldown_seconds: int = Field(
        30,
        ge=5,
        description="Cooldown period before recovery",
    )
    metrics_window_size: int = Field(
        100,
        ge=10,
        description="Window size for metrics calculation",
    )
    auto_recovery_enabled: bool = Field(
        True,
        description="Enable automatic recovery",
    )


# =============================================================================
# RESULT MODELS
# =============================================================================


class BackpressureStatus(BaseModel):
    """Current backpressure status."""

    backpressure_detected: bool = Field(
        ...,
        description="Whether backpressure is detected",
    )
    level: BackpressureLevel = Field(..., description="Backpressure level")
    queue_depth: int = Field(..., description="Current queue depth")
    queue_capacity: int = Field(0, description="Queue capacity if known")
    fill_ratio: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Queue fill ratio",
    )
    mode: OperationMode = Field(
        OperationMode.NORMAL,
        description="Current operation mode",
    )
    active_strategy: Optional[BackpressureStrategy] = Field(
        None,
        description="Currently active strategy",
    )
    items_shed: int = Field(0, ge=0, description="Items shed in current period")
    items_processed: int = Field(0, ge=0, description="Items processed in period")
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Detection timestamp",
    )


class BackpressureResult(BaseModel):
    """Result of applying backpressure."""

    success: bool = Field(..., description="Whether backpressure was applied")
    strategy: BackpressureStrategy = Field(
        ...,
        description="Strategy applied",
    )
    previous_mode: OperationMode = Field(
        ...,
        description="Previous operation mode",
    )
    current_mode: OperationMode = Field(
        ...,
        description="Current operation mode",
    )
    rate_limit_applied: Optional[float] = Field(
        None,
        description="Rate limit if applied",
    )
    sample_rate_applied: Optional[float] = Field(
        None,
        description="Sample rate if applied",
    )
    message: str = Field("", description="Result message")
    applied_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Application timestamp",
    )


class LoadShedResult(BaseModel):
    """Result of load shedding operation."""

    success: bool = Field(..., description="Whether load was shed successfully")
    priority: LoadPriority = Field(
        ...,
        description="Priority level that was shed",
    )
    items_shed: int = Field(0, ge=0, description="Number of items shed")
    items_retained: int = Field(0, ge=0, description="Items retained")
    shed_ratio: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of items shed",
    )
    queue_depth_before: int = Field(0, ge=0, description="Queue depth before shed")
    queue_depth_after: int = Field(0, ge=0, description="Queue depth after shed")
    shed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Shed timestamp",
    )


class ResumeResult(BaseModel):
    """Result of resuming normal operation."""

    success: bool = Field(..., description="Whether resume was successful")
    previous_mode: OperationMode = Field(
        ...,
        description="Previous operation mode",
    )
    current_mode: OperationMode = Field(
        ...,
        description="Current operation mode",
    )
    recovery_duration_seconds: float = Field(
        0.0,
        ge=0.0,
        description="Time spent in degraded mode",
    )
    items_processed_during_degraded: int = Field(
        0,
        ge=0,
        description="Items processed during degraded mode",
    )
    items_shed_during_degraded: int = Field(
        0,
        ge=0,
        description="Items shed during degraded mode",
    )
    resumed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Resume timestamp",
    )


# =============================================================================
# BACKPRESSURE HANDLER IMPLEMENTATION
# =============================================================================


@dataclass
class HandlerMetrics:
    """Metrics for backpressure monitoring."""

    queue_depths: Deque[int] = field(default_factory=lambda: deque(maxlen=100))
    processing_rates: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    shed_counts: Deque[int] = field(default_factory=lambda: deque(maxlen=100))
    backpressure_events: int = 0
    total_items_shed: int = 0
    total_items_processed: int = 0
    degraded_time_seconds: float = 0.0
    last_degraded_start: Optional[datetime] = None

    def record_queue_depth(self, depth: int) -> None:
        """Record queue depth observation."""
        self.queue_depths.append(depth)

    def record_processing_rate(self, rate: float) -> None:
        """Record processing rate."""
        self.processing_rates.append(rate)

    def record_shed(self, count: int) -> None:
        """Record items shed."""
        self.shed_counts.append(count)
        self.total_items_shed += count

    def get_avg_queue_depth(self) -> float:
        """Get average queue depth."""
        if not self.queue_depths:
            return 0.0
        return sum(self.queue_depths) / len(self.queue_depths)

    def get_avg_processing_rate(self) -> float:
        """Get average processing rate."""
        if not self.processing_rates:
            return 0.0
        return sum(self.processing_rates) / len(self.processing_rates)


class BackpressureHandler:
    """
    Handler for backpressure detection and load shedding.

    This handler monitors queue depths and applies backpressure strategies
    to prevent system overload during high-load periods.

    Example:
        >>> config = BackpressureConfig()
        >>> handler = BackpressureHandler(config)
        >>> status = handler.detect_backpressure(1500)
        >>> if status.backpressure_detected:
        ...     result = handler.apply_backpressure("adaptive")
    """

    def __init__(self, config: Optional[BackpressureConfig] = None) -> None:
        """
        Initialize BackpressureHandler.

        Args:
            config: Handler configuration
        """
        self.config = config or BackpressureConfig()
        self._mode = OperationMode.NORMAL
        self._active_strategy: Optional[BackpressureStrategy] = None
        self._current_rate_limit: float = 1.0
        self._current_sample_rate: float = 1.0
        self._last_backpressure_time: Optional[datetime] = None
        self._degraded_start_time: Optional[datetime] = None
        self._queue_depth = 0
        self._queue_capacity = self.config.critical_threshold * 2

        self.metrics = HandlerMetrics()

        # Priority queue simulation
        self._priority_queues: Dict[LoadPriority, List[Any]] = {
            LoadPriority.LOW: [],
            LoadPriority.NORMAL: [],
            LoadPriority.HIGH: [],
            LoadPriority.CRITICAL: [],
        }

        logger.info(
            f"BackpressureHandler initialized: "
            f"handler_id={self.config.handler_id}, "
            f"thresholds=({self.config.low_threshold}, "
            f"{self.config.moderate_threshold}, "
            f"{self.config.high_threshold}, "
            f"{self.config.critical_threshold})"
        )

    def detect_backpressure(
        self,
        queue_depth: int,
    ) -> BackpressureStatus:
        """
        Detect backpressure based on queue depth.

        Args:
            queue_depth: Current queue depth

        Returns:
            BackpressureStatus with detection results
        """
        self._queue_depth = queue_depth
        self.metrics.record_queue_depth(queue_depth)

        fill_ratio = queue_depth / self._queue_capacity if self._queue_capacity > 0 else 0.0

        # Determine backpressure level
        if queue_depth >= self.config.critical_threshold:
            level = BackpressureLevel.CRITICAL
            detected = True
        elif queue_depth >= self.config.high_threshold:
            level = BackpressureLevel.HIGH
            detected = True
        elif queue_depth >= self.config.moderate_threshold:
            level = BackpressureLevel.MODERATE
            detected = True
        elif queue_depth >= self.config.low_threshold:
            level = BackpressureLevel.LOW
            detected = True
        else:
            level = BackpressureLevel.NONE
            detected = False

        if detected and self._last_backpressure_time is None:
            self._last_backpressure_time = datetime.now(timezone.utc)
            self.metrics.backpressure_events += 1

            logger.warning(
                f"Backpressure detected: level={level.value}, "
                f"queue_depth={queue_depth}, fill_ratio={fill_ratio:.2%}"
            )

        return BackpressureStatus(
            backpressure_detected=detected,
            level=level,
            queue_depth=queue_depth,
            queue_capacity=self._queue_capacity,
            fill_ratio=fill_ratio,
            mode=self._mode,
            active_strategy=self._active_strategy,
            items_shed=self.metrics.total_items_shed,
            items_processed=self.metrics.total_items_processed,
        )

    def apply_backpressure(
        self,
        strategy: str,
    ) -> BackpressureResult:
        """
        Apply backpressure strategy.

        Args:
            strategy: Strategy to apply (rate_limit, sampling, etc.)

        Returns:
            BackpressureResult with application status
        """
        try:
            bp_strategy = BackpressureStrategy(strategy)
        except ValueError:
            bp_strategy = self.config.default_strategy

        previous_mode = self._mode
        message = ""
        rate_limit = None
        sample_rate = None

        if bp_strategy == BackpressureStrategy.RATE_LIMIT:
            self._current_rate_limit = self.config.rate_limit_factor
            self._mode = OperationMode.DEGRADED
            rate_limit = self._current_rate_limit
            message = f"Rate limited to {self._current_rate_limit:.0%}"

        elif bp_strategy == BackpressureStrategy.SAMPLING:
            self._current_sample_rate = self.config.sample_rate
            self._mode = OperationMode.DEGRADED
            sample_rate = self._current_sample_rate
            message = f"Sampling at {self._current_sample_rate:.0%}"

        elif bp_strategy == BackpressureStrategy.PRIORITY_SHED:
            self._mode = OperationMode.DEGRADED
            message = "Priority-based load shedding enabled"

        elif bp_strategy == BackpressureStrategy.PAUSE:
            self._mode = OperationMode.PAUSED
            message = "Processing paused"

        elif bp_strategy == BackpressureStrategy.REJECT:
            self._mode = OperationMode.EMERGENCY
            message = "Rejecting new items"

        elif bp_strategy == BackpressureStrategy.ADAPTIVE:
            # Adaptive strategy based on current level
            fill_ratio = self._queue_depth / self._queue_capacity

            if fill_ratio > 0.9:
                self._current_rate_limit = 0.2
                self._current_sample_rate = 0.3
            elif fill_ratio > 0.7:
                self._current_rate_limit = 0.4
                self._current_sample_rate = 0.5
            elif fill_ratio > 0.5:
                self._current_rate_limit = 0.6
                self._current_sample_rate = 0.7
            else:
                self._current_rate_limit = 0.8
                self._current_sample_rate = 0.8

            self._mode = OperationMode.DEGRADED
            rate_limit = self._current_rate_limit
            sample_rate = self._current_sample_rate
            message = (
                f"Adaptive: rate={self._current_rate_limit:.0%}, "
                f"sample={self._current_sample_rate:.0%}"
            )

        self._active_strategy = bp_strategy

        if self._mode == OperationMode.DEGRADED and self._degraded_start_time is None:
            self._degraded_start_time = datetime.now(timezone.utc)

        logger.info(
            f"Backpressure applied: strategy={bp_strategy.value}, "
            f"mode={self._mode.value}, {message}"
        )

        return BackpressureResult(
            success=True,
            strategy=bp_strategy,
            previous_mode=previous_mode,
            current_mode=self._mode,
            rate_limit_applied=rate_limit,
            sample_rate_applied=sample_rate,
            message=message,
        )

    def shed_load(
        self,
        priority: str,
    ) -> LoadShedResult:
        """
        Shed load at or below the specified priority level.

        Args:
            priority: Priority level to shed (items at this level and below)

        Returns:
            LoadShedResult with shedding results
        """
        try:
            shed_priority = LoadPriority(priority)
        except ValueError:
            shed_priority = LoadPriority.LOW

        queue_depth_before = sum(len(q) for q in self._priority_queues.values())

        # Determine which priorities to shed
        priority_order = [
            LoadPriority.LOW,
            LoadPriority.NORMAL,
            LoadPriority.HIGH,
            LoadPriority.CRITICAL,
        ]
        shed_index = priority_order.index(shed_priority)
        priorities_to_shed = priority_order[:shed_index + 1]

        items_shed = 0
        items_retained = 0

        for p in priorities_to_shed:
            count = len(self._priority_queues[p])
            items_shed += count
            self._priority_queues[p].clear()

        for p in priority_order[shed_index + 1:]:
            items_retained += len(self._priority_queues[p])

        queue_depth_after = sum(len(q) for q in self._priority_queues.values())

        self.metrics.record_shed(items_shed)

        shed_ratio = items_shed / queue_depth_before if queue_depth_before > 0 else 0.0

        logger.warning(
            f"Load shed: priority={shed_priority.value}, "
            f"items_shed={items_shed}, items_retained={items_retained}"
        )

        return LoadShedResult(
            success=True,
            priority=shed_priority,
            items_shed=items_shed,
            items_retained=items_retained,
            shed_ratio=shed_ratio,
            queue_depth_before=queue_depth_before,
            queue_depth_after=queue_depth_after,
        )

    def resume_normal_operation(self) -> ResumeResult:
        """
        Resume normal operation after backpressure.

        Returns:
            ResumeResult with recovery status
        """
        previous_mode = self._mode

        # Check if we can safely resume
        fill_ratio = self._queue_depth / self._queue_capacity if self._queue_capacity > 0 else 0.0

        if fill_ratio > self.config.recovery_threshold:
            logger.warning(
                f"Cannot resume: fill_ratio={fill_ratio:.2%} > "
                f"recovery_threshold={self.config.recovery_threshold:.2%}"
            )
            return ResumeResult(
                success=False,
                previous_mode=previous_mode,
                current_mode=self._mode,
                recovery_duration_seconds=0.0,
                items_processed_during_degraded=0,
                items_shed_during_degraded=0,
            )

        # Calculate time in degraded mode
        recovery_duration = 0.0
        if self._degraded_start_time:
            recovery_duration = (
                datetime.now(timezone.utc) - self._degraded_start_time
            ).total_seconds()
            self.metrics.degraded_time_seconds += recovery_duration

        # Reset state
        self._mode = OperationMode.NORMAL
        self._active_strategy = None
        self._current_rate_limit = 1.0
        self._current_sample_rate = 1.0
        self._last_backpressure_time = None
        self._degraded_start_time = None

        logger.info(
            f"Resumed normal operation after {recovery_duration:.1f}s in degraded mode"
        )

        return ResumeResult(
            success=True,
            previous_mode=previous_mode,
            current_mode=self._mode,
            recovery_duration_seconds=recovery_duration,
            items_processed_during_degraded=self.metrics.total_items_processed,
            items_shed_during_degraded=self.metrics.total_items_shed,
        )

    def should_process(self, priority: LoadPriority = LoadPriority.NORMAL) -> bool:
        """
        Determine if an item should be processed based on current backpressure.

        Args:
            priority: Item priority

        Returns:
            True if item should be processed
        """
        if self._mode == OperationMode.NORMAL:
            return True

        if self._mode == OperationMode.PAUSED:
            return False

        if self._mode == OperationMode.EMERGENCY:
            return priority == LoadPriority.CRITICAL

        # Degraded mode - apply rate limiting and sampling
        import random

        # Priority-based decision
        priority_thresholds = {
            LoadPriority.CRITICAL: 1.0,
            LoadPriority.HIGH: 0.9,
            LoadPriority.NORMAL: self._current_sample_rate,
            LoadPriority.LOW: self._current_sample_rate * 0.5,
        }

        threshold = priority_thresholds.get(priority, self._current_sample_rate)

        return random.random() < threshold * self._current_rate_limit

    def enqueue(
        self,
        item: Any,
        priority: LoadPriority = LoadPriority.NORMAL,
    ) -> bool:
        """
        Enqueue an item with priority.

        Args:
            item: Item to enqueue
            priority: Item priority

        Returns:
            True if item was enqueued
        """
        if self._mode == OperationMode.EMERGENCY and priority != LoadPriority.CRITICAL:
            return False

        self._priority_queues[priority].append(item)
        return True

    def set_queue_capacity(self, capacity: int) -> None:
        """Set the queue capacity."""
        self._queue_capacity = capacity

    def check_auto_recovery(self) -> Optional[ResumeResult]:
        """
        Check if automatic recovery should be triggered.

        Returns:
            ResumeResult if recovery triggered, None otherwise
        """
        if not self.config.auto_recovery_enabled:
            return None

        if self._mode == OperationMode.NORMAL:
            return None

        fill_ratio = self._queue_depth / self._queue_capacity if self._queue_capacity > 0 else 0.0

        if fill_ratio > self.config.recovery_threshold:
            return None

        # Check cooldown
        if self._degraded_start_time:
            elapsed = (datetime.now(timezone.utc) - self._degraded_start_time).total_seconds()
            if elapsed < self.config.recovery_cooldown_seconds:
                return None

        return self.resume_normal_operation()

    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            "handler_id": self.config.handler_id,
            "mode": self._mode.value,
            "active_strategy": self._active_strategy.value if self._active_strategy else None,
            "current_rate_limit": self._current_rate_limit,
            "current_sample_rate": self._current_sample_rate,
            "queue_depth": self._queue_depth,
            "queue_capacity": self._queue_capacity,
            "fill_ratio": self._queue_depth / self._queue_capacity if self._queue_capacity > 0 else 0.0,
            "backpressure_events": self.metrics.backpressure_events,
            "total_items_shed": self.metrics.total_items_shed,
            "total_items_processed": self.metrics.total_items_processed,
            "degraded_time_seconds": self.metrics.degraded_time_seconds,
            "avg_queue_depth": self.metrics.get_avg_queue_depth(),
            "avg_processing_rate": self.metrics.get_avg_processing_rate(),
        }

    @property
    def mode(self) -> OperationMode:
        """Return current operation mode."""
        return self._mode

    @property
    def is_degraded(self) -> bool:
        """Check if handler is in degraded mode."""
        return self._mode in (OperationMode.DEGRADED, OperationMode.EMERGENCY)

    @property
    def active_strategy(self) -> Optional[BackpressureStrategy]:
        """Return active backpressure strategy."""
        return self._active_strategy
