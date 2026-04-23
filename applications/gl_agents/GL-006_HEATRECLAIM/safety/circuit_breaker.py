# -*- coding: utf-8 -*-
"""
GL-006 HEATRECLAIM - Dynamic Circuit Breaker Pattern Implementation

This module implements an adaptive/dynamic circuit breaker pattern for resilient
external service calls in the HEATRECLAIM Heat Exchanger Network optimizer.
Provides fault tolerance with self-tuning thresholds based on system behavior.

Dynamic Circuit Breaker Features:
1. Adaptive failure thresholds based on historical performance
2. Exponential backoff recovery with jitter
3. Load-shedding during high contention
4. Health score tracking for gradual degradation
5. Self-healing with performance-based threshold adjustment

State Diagram:
    CLOSED ----[failure_threshold exceeded]----> OPEN
       ^                                            |
       |                                            v
       +---[success_threshold met]<--- HALF_OPEN <---[recovery_timeout + jitter]
                                           |
       +---[failure]<----------------------+

Reference Standards:
    - IEC 61508 (Functional Safety) - Fault Tolerance Requirements
    - IEC 61511 (SIS for Process Industries) - System Availability
    - ASME PTC 4.3/4.4 (Heat Exchangers) - Safety Requirements
    - API 660 (Shell-and-Tube Heat Exchangers)
    - ISO 14414 (Pump System Energy Assessment)

FAIL-SAFE Design:
When in doubt, the circuit breaker OPENS (fails safe). This prevents
potentially dangerous HEN configurations when external systems are unreliable.

Example:
    >>> from safety.circuit_breaker import DynamicCircuitBreaker, circuit_protected
    >>>
    >>> # Dynamic circuit breaker with adaptive thresholds
    >>> breaker = DynamicCircuitBreaker(
    ...     name="opcua_connector",
    ...     base_failure_threshold=5,
    ...     min_failure_threshold=2,
    ...     max_failure_threshold=10,
    ... )
    >>>
    >>> async with breaker.protect():
    ...     result = await opcua_client.read_heat_exchanger_temps()

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import random
import statistics
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# ENUMERATIONS
# =============================================================================


class CircuitBreakerState(str, Enum):
    """
    Circuit breaker state enumeration.

    States follow the standard circuit breaker pattern with extensions:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Circuit is tripped, requests fail immediately (FAIL-SAFE)
    - HALF_OPEN: Recovery testing, limited requests allowed
    - FORCED_OPEN: Manually forced open for maintenance/testing
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    FORCED_OPEN = "forced_open"


class CircuitBreakerEvent(str, Enum):
    """Events emitted by circuit breaker state transitions."""

    FAILURE_RECORDED = "failure_recorded"
    SUCCESS_RECORDED = "success_recorded"
    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_HALF_OPENED = "circuit_half_opened"
    CIRCUIT_CLOSED = "circuit_closed"
    CIRCUIT_FORCED_OPEN = "circuit_forced_open"
    CALL_BLOCKED = "call_blocked"
    CALL_ALLOWED = "call_allowed"
    THRESHOLD_ADJUSTED = "threshold_adjusted"
    HEALTH_SCORE_UPDATED = "health_score_updated"
    LOAD_SHED_ACTIVATED = "load_shed_activated"
    LOAD_SHED_DEACTIVATED = "load_shed_deactivated"


class HealthLevel(str, Enum):
    """Health level classification for adaptive behavior."""

    EXCELLENT = "excellent"  # 95-100% success rate
    GOOD = "good"  # 85-95%
    FAIR = "fair"  # 70-85%
    POOR = "poor"  # 50-70%
    CRITICAL = "critical"  # <50%


# =============================================================================
# EXCEPTIONS
# =============================================================================


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""

    def __init__(
        self,
        message: str,
        breaker_name: str,
        state: CircuitBreakerState,
    ) -> None:
        super().__init__(message)
        self.breaker_name = breaker_name
        self.state = state


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is open and calls are blocked (FAIL-SAFE behavior)."""

    def __init__(
        self,
        breaker_name: str,
        time_until_half_open: float,
        health_score: float = 0.0,
    ) -> None:
        message = (
            f"Circuit breaker '{breaker_name}' is OPEN (fail-safe). "
            f"Retry in {time_until_half_open:.1f}s. Health: {health_score:.1%}"
        )
        super().__init__(message, breaker_name, CircuitBreakerState.OPEN)
        self.time_until_half_open = time_until_half_open
        self.health_score = health_score


class CircuitHalfOpenError(CircuitBreakerError):
    """Raised when circuit is half-open and max test calls exceeded."""

    def __init__(self, breaker_name: str) -> None:
        message = (
            f"Circuit breaker '{breaker_name}' is HALF_OPEN. "
            f"Maximum test calls in progress."
        )
        super().__init__(message, breaker_name, CircuitBreakerState.HALF_OPEN)


class LoadShedError(CircuitBreakerError):
    """Raised when load shedding is active and call is rejected."""

    def __init__(self, breaker_name: str, shed_ratio: float) -> None:
        message = (
            f"Circuit breaker '{breaker_name}' load shedding active. "
            f"Rejection ratio: {shed_ratio:.1%}"
        )
        super().__init__(message, breaker_name, CircuitBreakerState.CLOSED)
        self.shed_ratio = shed_ratio


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class DynamicCircuitBreakerConfig(BaseModel):
    """
    Configuration for dynamic/adaptive circuit breaker.

    Attributes:
        name: Unique identifier for this circuit breaker
        base_failure_threshold: Starting failure threshold
        min_failure_threshold: Minimum threshold (for degraded systems)
        max_failure_threshold: Maximum threshold (for healthy systems)
        base_recovery_timeout_seconds: Base recovery timeout
        max_recovery_timeout_seconds: Max timeout with exponential backoff
        jitter_factor: Random jitter factor for recovery (0-1)
        health_window_size: Window size for health score calculation
        threshold_adjustment_interval: Seconds between threshold adjustments
        load_shed_threshold: Health score below which load shedding activates
        load_shed_max_ratio: Maximum load shed ratio (0-1)

    Reference:
        - Netflix Hystrix adaptive patterns
        - AWS Circuit Breaker patterns
        - IEC 61508 safety availability targets
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique circuit breaker identifier"
    )
    base_failure_threshold: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Base failure threshold"
    )
    min_failure_threshold: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum failure threshold (stricter)"
    )
    max_failure_threshold: int = Field(
        default=15,
        ge=5,
        le=100,
        description="Maximum failure threshold (more lenient)"
    )
    base_recovery_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Base recovery timeout"
    )
    max_recovery_timeout_seconds: float = Field(
        default=300.0,
        ge=30.0,
        le=600.0,
        description="Maximum recovery timeout with backoff"
    )
    jitter_factor: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Jitter factor for recovery timing"
    )
    half_open_max_calls: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max concurrent calls in half-open state"
    )
    success_threshold: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Successes needed to close circuit"
    )
    health_window_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Window size for health calculation"
    )
    threshold_adjustment_interval: float = Field(
        default=60.0,
        ge=10.0,
        le=600.0,
        description="Seconds between threshold adjustments"
    )
    load_shed_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Health score below which load shedding activates"
    )
    load_shed_max_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=0.9,
        description="Maximum load shed ratio"
    )
    slow_call_duration_threshold_ms: float = Field(
        default=5000.0,
        ge=100.0,
        le=60000.0,
        description="Slow call threshold in milliseconds"
    )
    enabled: bool = Field(
        default=True,
        description="Enable/disable circuit breaker"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name contains only safe characters."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Name must contain only alphanumeric, underscore, or hyphen")
        return v


class CircuitBreakerMetrics(BaseModel):
    """Metrics for circuit breaker monitoring and dashboards."""

    name: str = Field(..., description="Circuit breaker name")
    state: CircuitBreakerState = Field(..., description="Current state")
    health_score: float = Field(default=1.0, ge=0.0, le=1.0)
    health_level: HealthLevel = Field(default=HealthLevel.EXCELLENT)
    current_failure_threshold: int = Field(default=5, ge=1)
    current_recovery_timeout: float = Field(default=30.0, ge=1.0)
    failure_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    consecutive_successes: int = Field(default=0, ge=0)
    open_count: int = Field(default=0, ge=0, description="Times circuit opened")
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    calls_blocked: int = Field(default=0, ge=0)
    calls_load_shed: int = Field(default=0, ge=0)
    total_calls: int = Field(default=0, ge=0)
    avg_call_duration_ms: float = Field(default=0.0, ge=0.0)
    load_shed_active: bool = Field(default=False)
    load_shed_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failure_count / self.total_calls

    @property
    def availability(self) -> float:
        """Calculate availability (success rate)."""
        return 1.0 - self.failure_rate

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        content = (
            f"{self.name}|{self.state.value}|{self.health_score:.4f}|"
            f"{self.failure_count}|{self.success_count}|"
            f"{self.state_changed_at.isoformat()}"
        )
        return hashlib.sha256(content.encode()).hexdigest()


class CircuitBreakerAuditRecord(BaseModel):
    """
    Audit record for circuit breaker state changes.

    Supports IEC 61508 audit trail requirements for HEN safety systems.
    """

    record_id: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    breaker_name: str = Field(..., description="Circuit breaker name")
    event: CircuitBreakerEvent = Field(..., description="Event type")
    previous_state: Optional[CircuitBreakerState] = None
    new_state: CircuitBreakerState = Field(..., description="New state")
    health_score: float = Field(default=1.0, ge=0.0, le=1.0)
    failure_threshold: int = Field(default=5, ge=1)
    error_message: Optional[str] = Field(default=None, max_length=500)
    context: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            content = (
                f"{self.record_id}|{self.timestamp.isoformat()}|"
                f"{self.breaker_name}|{self.event.value}|{self.new_state.value}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# CALL RECORD AND SLIDING WINDOW
# =============================================================================


@dataclass
class CallRecord:
    """Record of a single call for health calculation."""

    timestamp: float
    duration_ms: float
    success: bool
    error: Optional[str] = None


class AdaptiveSlidingWindow:
    """
    Thread-safe sliding window with adaptive health scoring.

    Provides:
    - Weighted health score (recent calls matter more)
    - Trend detection for proactive threshold adjustment
    - Statistical analysis for anomaly detection
    """

    def __init__(self, size: int = 100) -> None:
        self._size = size
        self._records: List[CallRecord] = []
        self._lock = threading.Lock()

    def record(self, call_record: CallRecord) -> None:
        """Add a record to the window."""
        with self._lock:
            self._records.append(call_record)
            if len(self._records) > self._size:
                self._records.pop(0)

    def get_health_score(self) -> float:
        """
        Calculate weighted health score.

        Recent calls have higher weight using exponential decay.
        """
        with self._lock:
            if not self._records:
                return 1.0

            # Exponential decay weights (more recent = higher weight)
            n = len(self._records)
            weights = [1.1 ** (i - n + 1) for i in range(n)]
            total_weight = sum(weights)

            # Calculate weighted success rate
            weighted_successes = sum(
                w for w, r in zip(weights, self._records) if r.success
            )

            return weighted_successes / total_weight

    def get_health_level(self) -> HealthLevel:
        """Classify health score into levels."""
        score = self.get_health_score()
        if score >= 0.95:
            return HealthLevel.EXCELLENT
        elif score >= 0.85:
            return HealthLevel.GOOD
        elif score >= 0.70:
            return HealthLevel.FAIR
        elif score >= 0.50:
            return HealthLevel.POOR
        else:
            return HealthLevel.CRITICAL

    def get_failure_rate(self) -> float:
        """Calculate failure rate from window."""
        with self._lock:
            if not self._records:
                return 0.0
            failures = sum(1 for r in self._records if not r.success)
            return failures / len(self._records)

    def get_trend(self) -> float:
        """
        Calculate trend direction (-1 to 1).

        Positive = improving, Negative = degrading
        """
        with self._lock:
            if len(self._records) < 10:
                return 0.0

            # Compare first half vs second half
            mid = len(self._records) // 2
            first_half = self._records[:mid]
            second_half = self._records[mid:]

            first_rate = sum(1 for r in first_half if r.success) / len(first_half)
            second_rate = sum(1 for r in second_half if r.success) / len(second_half)

            return second_rate - first_rate

    def get_avg_duration(self) -> float:
        """Calculate average call duration."""
        with self._lock:
            if not self._records:
                return 0.0
            return sum(r.duration_ms for r in self._records) / len(self._records)

    def get_duration_std(self) -> float:
        """Calculate standard deviation of call durations."""
        with self._lock:
            if len(self._records) < 2:
                return 0.0
            durations = [r.duration_ms for r in self._records]
            return statistics.stdev(durations)

    def clear(self) -> None:
        """Clear all records."""
        with self._lock:
            self._records.clear()

    @property
    def count(self) -> int:
        """Get current record count."""
        with self._lock:
            return len(self._records)


# =============================================================================
# DYNAMIC CIRCUIT BREAKER IMPLEMENTATION
# =============================================================================


class DynamicCircuitBreaker:
    """
    Adaptive/Dynamic Circuit Breaker for GL-006 HEATRECLAIM.

    Implements circuit breaker pattern with self-tuning capabilities:
    - Adaptive failure thresholds based on system health
    - Exponential backoff with jitter for recovery
    - Load shedding during degradation
    - Health-based threshold adjustment

    Design follows:
    - IEC 61508 fault tolerance requirements
    - Netflix Hystrix adaptive patterns
    - AWS Circuit Breaker best practices
    - ASME PTC 4.3/4.4 HEN safety requirements

    Example:
        >>> breaker = DynamicCircuitBreaker(
        ...     name="hen_optimizer_api",
        ...     base_failure_threshold=5,
        ... )
        >>>
        >>> async with breaker.protect():
        ...     result = await optimize_heat_network()
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        name: str,
        base_failure_threshold: int = 5,
        min_failure_threshold: int = 2,
        max_failure_threshold: int = 15,
        base_recovery_timeout_seconds: float = 30.0,
        max_recovery_timeout_seconds: float = 300.0,
        config: Optional[DynamicCircuitBreakerConfig] = None,
        event_callback: Optional[Callable[[CircuitBreakerAuditRecord], None]] = None,
    ) -> None:
        """
        Initialize dynamic circuit breaker.

        Args:
            name: Unique identifier
            base_failure_threshold: Starting failure threshold
            min_failure_threshold: Minimum (stricter) threshold
            max_failure_threshold: Maximum (lenient) threshold
            base_recovery_timeout_seconds: Base recovery timeout
            max_recovery_timeout_seconds: Max timeout with backoff
            config: Optional full configuration
            event_callback: Optional callback for events
        """
        if config:
            self._config = config
        else:
            self._config = DynamicCircuitBreakerConfig(
                name=name,
                base_failure_threshold=base_failure_threshold,
                min_failure_threshold=min_failure_threshold,
                max_failure_threshold=max_failure_threshold,
                base_recovery_timeout_seconds=base_recovery_timeout_seconds,
                max_recovery_timeout_seconds=max_recovery_timeout_seconds,
            )

        self._state = CircuitBreakerState.CLOSED
        self._state_lock = threading.Lock()
        self._state_changed_at = time.monotonic()

        # Dynamic thresholds
        self._current_failure_threshold = self._config.base_failure_threshold
        self._current_recovery_timeout = self._config.base_recovery_timeout_seconds
        self._consecutive_opens = 0  # For exponential backoff

        # Counters
        self._failure_count = 0
        self._success_count = 0
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._calls_blocked = 0
        self._calls_load_shed = 0
        self._total_calls = 0
        self._half_open_calls = 0
        self._open_count = 0

        # Timestamps
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
        self._opened_at: Optional[float] = None
        self._last_threshold_adjustment: float = time.monotonic()

        # Health tracking
        self._window = AdaptiveSlidingWindow(self._config.health_window_size)
        self._load_shed_active = False
        self._load_shed_ratio = 0.0

        # Event callback
        self._event_callback = event_callback

        # Audit trail
        self._audit_records: List[CircuitBreakerAuditRecord] = []

        logger.info(
            "DynamicCircuitBreaker initialized: name=%s, base_threshold=%d, "
            "base_timeout=%.1fs",
            self._config.name,
            self._config.base_failure_threshold,
            self._config.base_recovery_timeout_seconds,
        )

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def state(self) -> CircuitBreakerState:
        with self._state_lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitBreakerState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state in (CircuitBreakerState.OPEN, CircuitBreakerState.FORCED_OPEN)

    @property
    def is_half_open(self) -> bool:
        return self.state == CircuitBreakerState.HALF_OPEN

    @property
    def health_score(self) -> float:
        return self._window.get_health_score()

    @property
    def health_level(self) -> HealthLevel:
        return self._window.get_health_level()

    @property
    def current_failure_threshold(self) -> int:
        return self._current_failure_threshold

    @property
    def current_recovery_timeout(self) -> float:
        return self._current_recovery_timeout

    @property
    def config(self) -> DynamicCircuitBreakerConfig:
        return self._config

    # =========================================================================
    # ADAPTIVE THRESHOLD MANAGEMENT
    # =========================================================================

    def _adjust_thresholds(self) -> None:
        """
        Dynamically adjust thresholds based on health score.

        Called periodically to tune circuit breaker behavior:
        - High health = more lenient thresholds (allow more failures)
        - Low health = stricter thresholds (open faster)
        """
        now = time.monotonic()
        if now - self._last_threshold_adjustment < self._config.threshold_adjustment_interval:
            return

        self._last_threshold_adjustment = now
        health = self.health_score
        trend = self._window.get_trend()

        # Calculate new threshold based on health
        threshold_range = self._config.max_failure_threshold - self._config.min_failure_threshold
        normalized_health = max(0, min(1, health))

        # Adjust threshold: higher health = higher threshold (more lenient)
        new_threshold = int(
            self._config.min_failure_threshold +
            (threshold_range * normalized_health)
        )

        # Consider trend: if degrading, be more conservative
        if trend < -0.1:  # Degrading
            new_threshold = max(
                self._config.min_failure_threshold,
                new_threshold - 1
            )
        elif trend > 0.1:  # Improving
            new_threshold = min(
                self._config.max_failure_threshold,
                new_threshold + 1
            )

        if new_threshold != self._current_failure_threshold:
            old_threshold = self._current_failure_threshold
            self._current_failure_threshold = new_threshold

            record = CircuitBreakerAuditRecord(
                breaker_name=self._config.name,
                event=CircuitBreakerEvent.THRESHOLD_ADJUSTED,
                new_state=self._state,
                health_score=health,
                failure_threshold=new_threshold,
                context={
                    "old_threshold": old_threshold,
                    "new_threshold": new_threshold,
                    "trend": trend,
                },
            )
            self._audit_records.append(record)

            logger.info(
                "DynamicCircuitBreaker threshold adjusted: %s -> %s (health=%.2f, trend=%.2f)",
                old_threshold,
                new_threshold,
                health,
                trend,
            )

    def _calculate_recovery_timeout(self) -> float:
        """
        Calculate recovery timeout with exponential backoff and jitter.

        Consecutive opens increase timeout exponentially up to max.
        Jitter prevents thundering herd on recovery.
        """
        # Exponential backoff: 2^consecutive_opens * base
        backoff = min(
            self._config.base_recovery_timeout_seconds * (2 ** self._consecutive_opens),
            self._config.max_recovery_timeout_seconds,
        )

        # Add jitter
        jitter = random.uniform(
            -self._config.jitter_factor * backoff,
            self._config.jitter_factor * backoff,
        )

        return max(
            self._config.base_recovery_timeout_seconds,
            min(backoff + jitter, self._config.max_recovery_timeout_seconds),
        )

    def _update_load_shedding(self) -> None:
        """Update load shedding state based on health score."""
        health = self.health_score

        if health < self._config.load_shed_threshold:
            if not self._load_shed_active:
                self._load_shed_active = True
                record = CircuitBreakerAuditRecord(
                    breaker_name=self._config.name,
                    event=CircuitBreakerEvent.LOAD_SHED_ACTIVATED,
                    new_state=self._state,
                    health_score=health,
                    failure_threshold=self._current_failure_threshold,
                )
                self._audit_records.append(record)
                logger.warning(
                    "DynamicCircuitBreaker load shedding activated: name=%s, health=%.2f",
                    self._config.name,
                    health,
                )

            # Calculate shed ratio based on how far below threshold
            ratio = (self._config.load_shed_threshold - health) / self._config.load_shed_threshold
            self._load_shed_ratio = min(ratio, self._config.load_shed_max_ratio)

        elif self._load_shed_active and health >= self._config.load_shed_threshold + 0.1:
            # Hysteresis: require 10% above threshold to deactivate
            self._load_shed_active = False
            self._load_shed_ratio = 0.0
            record = CircuitBreakerAuditRecord(
                breaker_name=self._config.name,
                event=CircuitBreakerEvent.LOAD_SHED_DEACTIVATED,
                new_state=self._state,
                health_score=health,
                failure_threshold=self._current_failure_threshold,
            )
            self._audit_records.append(record)
            logger.info(
                "DynamicCircuitBreaker load shedding deactivated: name=%s, health=%.2f",
                self._config.name,
                health,
            )

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def _transition_to(
        self,
        new_state: CircuitBreakerState,
        reason: str = "",
    ) -> None:
        """Transition to a new state (thread-safe)."""
        with self._state_lock:
            if self._state == new_state:
                return

            old_state = self._state
            self._state = new_state
            self._state_changed_at = time.monotonic()

            # State-specific actions
            if new_state == CircuitBreakerState.HALF_OPEN:
                self._half_open_calls = 0
                self._consecutive_successes = 0
            elif new_state == CircuitBreakerState.CLOSED:
                self._consecutive_failures = 0
                self._consecutive_opens = 0  # Reset backoff on successful close
                self._current_recovery_timeout = self._config.base_recovery_timeout_seconds
            elif new_state == CircuitBreakerState.OPEN:
                self._opened_at = time.monotonic()
                self._open_count += 1
                self._consecutive_opens += 1
                self._current_recovery_timeout = self._calculate_recovery_timeout()

            # Determine event type
            event_map = {
                CircuitBreakerState.OPEN: CircuitBreakerEvent.CIRCUIT_OPENED,
                CircuitBreakerState.HALF_OPEN: CircuitBreakerEvent.CIRCUIT_HALF_OPENED,
                CircuitBreakerState.CLOSED: CircuitBreakerEvent.CIRCUIT_CLOSED,
                CircuitBreakerState.FORCED_OPEN: CircuitBreakerEvent.CIRCUIT_FORCED_OPEN,
            }

            record = CircuitBreakerAuditRecord(
                breaker_name=self._config.name,
                event=event_map[new_state],
                previous_state=old_state,
                new_state=new_state,
                health_score=self.health_score,
                failure_threshold=self._current_failure_threshold,
                context={
                    "reason": reason,
                    "recovery_timeout": self._current_recovery_timeout,
                    "consecutive_opens": self._consecutive_opens,
                },
            )
            self._audit_records.append(record)

            if self._event_callback:
                try:
                    self._event_callback(record)
                except Exception as e:
                    logger.error("Event callback failed: %s", e)

            logger.warning(
                "DynamicCircuitBreaker state transition: %s -> %s (name=%s, reason=%s, timeout=%.1fs)",
                old_state.value,
                new_state.value,
                self._config.name,
                reason,
                self._current_recovery_timeout,
            )

    def _get_time_until_half_open(self) -> float:
        """Get seconds until circuit enters half-open state."""
        if self._state != CircuitBreakerState.OPEN:
            return 0.0
        if self._opened_at is None:
            return self._current_recovery_timeout
        elapsed = time.monotonic() - self._opened_at
        return max(0.0, self._current_recovery_timeout - elapsed)

    # =========================================================================
    # CALL RECORDING
    # =========================================================================

    def record_success(self, duration_ms: float = 0.0) -> None:
        """Record a successful call."""
        with self._state_lock:
            self._success_count += 1
            self._consecutive_successes += 1
            self._consecutive_failures = 0
            self._last_success_time = time.monotonic()
            self._total_calls += 1

            self._window.record(CallRecord(
                timestamp=time.monotonic(),
                duration_ms=duration_ms,
                success=True,
            ))

            # State transitions
            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._consecutive_successes >= self._config.success_threshold:
                    self._transition_to(
                        CircuitBreakerState.CLOSED,
                        reason=f"Success threshold reached ({self._consecutive_successes})"
                    )

        # Periodic adjustments (outside lock)
        self._adjust_thresholds()
        self._update_load_shedding()

    def record_failure(
        self,
        error: Optional[Exception] = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Record a failed call."""
        with self._state_lock:
            self._failure_count += 1
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._last_failure_time = time.monotonic()
            self._total_calls += 1

            error_msg = str(error) if error else None

            self._window.record(CallRecord(
                timestamp=time.monotonic(),
                duration_ms=duration_ms,
                success=False,
                error=error_msg,
            ))

            # State transitions
            if self._state == CircuitBreakerState.CLOSED:
                if self._consecutive_failures >= self._current_failure_threshold:
                    self._transition_to(
                        CircuitBreakerState.OPEN,
                        reason=f"Failure threshold reached ({self._consecutive_failures}/{self._current_failure_threshold})"
                    )

            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open reopens the circuit (FAIL-SAFE)
                self._transition_to(
                    CircuitBreakerState.OPEN,
                    reason="Failure in half-open state"
                )

            logger.warning(
                "DynamicCircuitBreaker failure: name=%s, consecutive=%d/%d, error=%s",
                self._config.name,
                self._consecutive_failures,
                self._current_failure_threshold,
                error_msg,
            )

        # Periodic adjustments (outside lock)
        self._adjust_thresholds()
        self._update_load_shedding()

    # =========================================================================
    # CALL EXECUTION
    # =========================================================================

    def allow_request(self) -> bool:
        """
        Check if a request should be allowed.

        Returns:
            True if request can proceed, False if blocked
        """
        if not self._config.enabled:
            return True

        with self._state_lock:
            # Check load shedding first (probabilistic rejection)
            if self._load_shed_active and self._state == CircuitBreakerState.CLOSED:
                if random.random() < self._load_shed_ratio:
                    self._calls_load_shed += 1
                    return False

            if self._state == CircuitBreakerState.CLOSED:
                return True

            elif self._state == CircuitBreakerState.OPEN:
                # Check if recovery timeout elapsed
                if self._opened_at is not None:
                    elapsed = time.monotonic() - self._opened_at
                    if elapsed >= self._current_recovery_timeout:
                        self._transition_to(
                            CircuitBreakerState.HALF_OPEN,
                            reason="Recovery timeout elapsed"
                        )
                        self._half_open_calls = 1
                        return True

                self._calls_blocked += 1
                return False

            elif self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls < self._config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            elif self._state == CircuitBreakerState.FORCED_OPEN:
                self._calls_blocked += 1
                return False

            return False

    @asynccontextmanager
    async def protect(self):
        """
        Async context manager for protected calls.

        FAIL-SAFE: If circuit is open, raises CircuitOpenError immediately.

        Usage:
            async with breaker.protect():
                result = await risky_operation()
        """
        if not self.allow_request():
            if self._state == CircuitBreakerState.OPEN:
                raise CircuitOpenError(
                    self._config.name,
                    self._get_time_until_half_open(),
                    self.health_score,
                )
            elif self._state == CircuitBreakerState.FORCED_OPEN:
                raise CircuitOpenError(
                    self._config.name,
                    float('inf'),
                    self.health_score,
                )
            elif self._load_shed_active:
                raise LoadShedError(self._config.name, self._load_shed_ratio)
            else:
                raise CircuitHalfOpenError(self._config.name)

        start_time = time.monotonic()
        try:
            yield
            duration_ms = (time.monotonic() - start_time) * 1000
            self.record_success(duration_ms)
        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.record_failure(e, duration_ms)
            raise

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        fallback: Optional[Callable[..., Awaitable[T]]] = None,
        **kwargs: Any,
    ) -> T:
        """Execute a function with circuit breaker protection."""
        try:
            async with self.protect():
                return await func(*args, **kwargs)
        except (CircuitOpenError, CircuitHalfOpenError, LoadShedError) as e:
            if fallback is not None:
                logger.info(
                    "DynamicCircuitBreaker fallback: name=%s",
                    self._config.name,
                )
                return await fallback(*args, **kwargs)
            raise

    # =========================================================================
    # MANUAL CONTROLS
    # =========================================================================

    def force_open(self, reason: str = "Manual force open") -> None:
        """Force circuit to open (for maintenance/testing)."""
        self._transition_to(CircuitBreakerState.FORCED_OPEN, reason=reason)

    def force_close(self, reason: str = "Manual force close") -> None:
        """Force circuit to close."""
        self._transition_to(CircuitBreakerState.CLOSED, reason=reason)

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._state_lock:
            old_state = self._state
            self._state = CircuitBreakerState.CLOSED
            self._state_changed_at = time.monotonic()
            self._current_failure_threshold = self._config.base_failure_threshold
            self._current_recovery_timeout = self._config.base_recovery_timeout_seconds
            self._consecutive_opens = 0
            self._failure_count = 0
            self._success_count = 0
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._calls_blocked = 0
            self._calls_load_shed = 0
            self._total_calls = 0
            self._half_open_calls = 0
            self._last_failure_time = None
            self._last_success_time = None
            self._opened_at = None
            self._load_shed_active = False
            self._load_shed_ratio = 0.0
            self._window.clear()

            if old_state != CircuitBreakerState.CLOSED:
                record = CircuitBreakerAuditRecord(
                    breaker_name=self._config.name,
                    event=CircuitBreakerEvent.CIRCUIT_CLOSED,
                    previous_state=old_state,
                    new_state=CircuitBreakerState.CLOSED,
                    context={"reason": "Manual reset"},
                )
                self._audit_records.append(record)

        logger.info("DynamicCircuitBreaker reset: name=%s", self._config.name)

    # =========================================================================
    # METRICS AND MONITORING
    # =========================================================================

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current circuit breaker metrics."""
        with self._state_lock:
            return CircuitBreakerMetrics(
                name=self._config.name,
                state=self._state,
                health_score=self._window.get_health_score(),
                health_level=self._window.get_health_level(),
                current_failure_threshold=self._current_failure_threshold,
                current_recovery_timeout=self._current_recovery_timeout,
                failure_count=self._failure_count,
                success_count=self._success_count,
                consecutive_failures=self._consecutive_failures,
                consecutive_successes=self._consecutive_successes,
                open_count=self._open_count,
                last_failure_time=(
                    datetime.fromtimestamp(self._last_failure_time, tz=timezone.utc)
                    if self._last_failure_time else None
                ),
                last_success_time=(
                    datetime.fromtimestamp(self._last_success_time, tz=timezone.utc)
                    if self._last_success_time else None
                ),
                state_changed_at=datetime.fromtimestamp(
                    self._state_changed_at, tz=timezone.utc
                ),
                calls_blocked=self._calls_blocked,
                calls_load_shed=self._calls_load_shed,
                total_calls=self._total_calls,
                avg_call_duration_ms=self._window.get_avg_duration(),
                load_shed_active=self._load_shed_active,
                load_shed_ratio=self._load_shed_ratio,
            )

    def get_audit_records(self, limit: int = 100) -> List[CircuitBreakerAuditRecord]:
        """Get recent audit records."""
        return list(reversed(self._audit_records[-limit:]))

    def __repr__(self) -> str:
        return (
            f"DynamicCircuitBreaker(name={self._config.name!r}, "
            f"state={self._state.value}, "
            f"health={self.health_score:.2f}, "
            f"threshold={self._current_failure_threshold})"
        )


# =============================================================================
# CIRCUIT BREAKER REGISTRY
# =============================================================================


class CircuitBreakerRegistry:
    """Thread-safe registry for managing multiple circuit breakers."""

    _instance: Optional["CircuitBreakerRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CircuitBreakerRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._breakers: Dict[str, DynamicCircuitBreaker] = {}
                cls._instance._registry_lock = threading.Lock()
            return cls._instance

    def register(self, breaker: DynamicCircuitBreaker) -> None:
        """Register a circuit breaker."""
        with self._registry_lock:
            self._breakers[breaker.name] = breaker

    def get(self, name: str) -> Optional[DynamicCircuitBreaker]:
        """Get circuit breaker by name."""
        with self._registry_lock:
            return self._breakers.get(name)

    def get_or_create(
        self,
        name: str,
        **kwargs: Any,
    ) -> DynamicCircuitBreaker:
        """Get existing or create new circuit breaker."""
        with self._registry_lock:
            if name in self._breakers:
                return self._breakers[name]
            breaker = DynamicCircuitBreaker(name=name, **kwargs)
            self._breakers[name] = breaker
            return breaker

    def get_all_metrics(self) -> Dict[str, CircuitBreakerMetrics]:
        """Get metrics for all circuit breakers."""
        with self._registry_lock:
            return {
                name: breaker.get_metrics()
                for name, breaker in self._breakers.items()
            }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._registry_lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def clear(self) -> None:
        """Clear all registered circuit breakers."""
        with self._registry_lock:
            self._breakers.clear()


# Global registry instance
_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str) -> Optional[DynamicCircuitBreaker]:
    """Get circuit breaker from global registry."""
    return _registry.get(name)


def get_or_create_circuit_breaker(
    name: str,
    **kwargs: Any,
) -> DynamicCircuitBreaker:
    """Get or create circuit breaker in global registry."""
    return _registry.get_or_create(name, **kwargs)


# =============================================================================
# DECORATOR
# =============================================================================


def circuit_protected(
    name: str,
    base_failure_threshold: int = 5,
    base_recovery_timeout_seconds: float = 30.0,
    fallback: Optional[Callable[..., Awaitable[Any]]] = None,
    **config_kwargs: Any,
) -> Callable[[F], F]:
    """
    Decorator to protect async functions with a dynamic circuit breaker.

    Example:
        >>> @circuit_protected(
        ...     name="hen_optimizer",
        ...     base_failure_threshold=5,
        ... )
        ... async def optimize_network(streams: List[Dict]) -> Dict:
        ...     return await optimizer.run(streams)
    """
    def decorator(func: F) -> F:
        breaker = get_or_create_circuit_breaker(
            name=name,
            base_failure_threshold=base_failure_threshold,
            base_recovery_timeout_seconds=base_recovery_timeout_seconds,
            **config_kwargs,
        )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await breaker.execute(
                func,
                *args,
                fallback=fallback,
                **kwargs,
            )

        wrapper._circuit_breaker = breaker  # type: ignore
        return wrapper  # type: ignore

    return decorator


# =============================================================================
# PRE-CONFIGURED CIRCUIT BREAKERS FOR HEATRECLAIM
# =============================================================================


class HeatReclaimCircuitBreakers:
    """
    Pre-configured circuit breakers for GL-006 HEATRECLAIM.

    Provides circuit breakers for HEN optimization external integrations:
    - OPC-UA for temperature/pressure sensors
    - MILP solver connections
    - Historian for historical thermal data
    - ERP for production scheduling
    """

    OPCUA_CONFIG = DynamicCircuitBreakerConfig(
        name="opcua_connector",
        base_failure_threshold=3,
        min_failure_threshold=2,
        max_failure_threshold=8,
        base_recovery_timeout_seconds=15.0,
        max_recovery_timeout_seconds=120.0,
    )

    MILP_SOLVER_CONFIG = DynamicCircuitBreakerConfig(
        name="milp_solver",
        base_failure_threshold=5,
        min_failure_threshold=3,
        max_failure_threshold=10,
        base_recovery_timeout_seconds=30.0,
        max_recovery_timeout_seconds=180.0,
        slow_call_duration_threshold_ms=30000.0,
    )

    HISTORIAN_CONFIG = DynamicCircuitBreakerConfig(
        name="historian_api",
        base_failure_threshold=5,
        min_failure_threshold=3,
        max_failure_threshold=12,
        base_recovery_timeout_seconds=60.0,
        max_recovery_timeout_seconds=300.0,
    )

    def __init__(
        self,
        event_callback: Optional[Callable[[CircuitBreakerAuditRecord], None]] = None,
    ) -> None:
        self._event_callback = event_callback

        self.opcua = DynamicCircuitBreaker(
            name="opcua_connector",
            config=self.OPCUA_CONFIG,
            event_callback=event_callback,
        )
        self.milp_solver = DynamicCircuitBreaker(
            name="milp_solver",
            config=self.MILP_SOLVER_CONFIG,
            event_callback=event_callback,
        )
        self.historian = DynamicCircuitBreaker(
            name="historian_api",
            config=self.HISTORIAN_CONFIG,
            event_callback=event_callback,
        )

        _registry.register(self.opcua)
        _registry.register(self.milp_solver)
        _registry.register(self.historian)

        logger.info("HeatReclaimCircuitBreakers initialized")

    def get_all_metrics(self) -> Dict[str, CircuitBreakerMetrics]:
        return {
            "opcua": self.opcua.get_metrics(),
            "milp_solver": self.milp_solver.get_metrics(),
            "historian": self.historian.get_metrics(),
        }

    def get_system_health(self) -> Dict[str, Any]:
        metrics = self.get_all_metrics()

        open_count = sum(
            1 for m in metrics.values()
            if m.state in (CircuitBreakerState.OPEN, CircuitBreakerState.FORCED_OPEN)
        )
        half_open_count = sum(
            1 for m in metrics.values()
            if m.state == CircuitBreakerState.HALF_OPEN
        )

        avg_health = sum(m.health_score for m in metrics.values()) / len(metrics)

        if open_count >= 2 or avg_health < 0.5:
            health = "critical"
        elif open_count >= 1 or avg_health < 0.7:
            health = "degraded"
        elif half_open_count >= 1 or avg_health < 0.85:
            health = "warning"
        else:
            health = "healthy"

        return {
            "health": health,
            "avg_health_score": avg_health,
            "open_circuits": open_count,
            "half_open_circuits": half_open_count,
            "circuits": {
                name: {"state": m.state.value, "health": m.health_score}
                for name, m in metrics.items()
            },
        }

    def reset_all(self) -> None:
        self.opcua.reset()
        self.milp_solver.reset()
        self.historian.reset()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "CircuitBreakerState",
    "CircuitBreakerEvent",
    "HealthLevel",
    # Exceptions
    "CircuitBreakerError",
    "CircuitOpenError",
    "CircuitHalfOpenError",
    "LoadShedError",
    # Models
    "DynamicCircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitBreakerAuditRecord",
    # Core Classes
    "DynamicCircuitBreaker",
    "CircuitBreakerRegistry",
    "HeatReclaimCircuitBreakers",
    # Functions
    "get_circuit_breaker",
    "get_or_create_circuit_breaker",
    "circuit_protected",
]
