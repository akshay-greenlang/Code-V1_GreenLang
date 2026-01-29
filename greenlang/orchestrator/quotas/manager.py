# -*- coding: utf-8 -*-
"""
QuotaManager - Namespace Concurrency Quota Management (FR-024)
==============================================================

Thread-safe multi-tenant quota management for resource isolation and
priority-based scheduling in the GreenLang Orchestrator.

This module implements:
- Per-namespace quota configuration and enforcement
- Real-time usage tracking with thread-safe operations
- Priority-weighted queue management across namespaces
- Admission control for runs and steps
- Queue timeout handling with event emission
- Comprehensive metrics for monitoring

Example:
    >>> manager = QuotaManager()
    >>> manager.load_from_yaml("config/namespace_quotas.yaml")
    >>>
    >>> # Admission control
    >>> if manager.can_submit_run("production"):
    ...     run_id = submit_run(...)
    ...     await manager.acquire_run_slot("production", run_id, priority=7)
    ...
    >>> # Release when done
    >>> await manager.release_run_slot("production", run_id)

Author: GreenLang Framework Team
Date: January 2026
GL-FOUND-X-001: FR-024 Namespace Concurrency Quotas
Status: Production Ready
"""

import asyncio
import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class QuotaEventType(str, Enum):
    """Event types emitted by QuotaManager."""

    CONCURRENCY_SLOT_ACQUIRED = "CONCURRENCY_SLOT_ACQUIRED"
    CONCURRENCY_SLOT_RELEASED = "CONCURRENCY_SLOT_RELEASED"
    CONCURRENCY_QUEUE_TIMEOUT = "CONCURRENCY_QUEUE_TIMEOUT"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    QUOTA_UPDATED = "QUOTA_UPDATED"
    NAMESPACE_CREATED = "NAMESPACE_CREATED"


DEFAULT_QUEUE_TIMEOUT_SECONDS = 300.0  # 5 minutes default queue timeout


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class QuotaConfig(BaseModel):
    """
    Configuration for namespace quotas.

    Defines resource limits and scheduling priority for a namespace.

    Attributes:
        max_concurrent_runs: Maximum simultaneous pipeline runs
        max_concurrent_steps: Maximum simultaneous steps across all runs
        max_queued_runs: Maximum runs waiting in queue
        priority_weight: Scheduling priority multiplier (higher = more priority)
        queue_timeout_seconds: How long a run can wait in queue before timeout
    """

    max_concurrent_runs: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Maximum concurrent runs for this namespace",
    )
    max_concurrent_steps: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum concurrent steps across all runs",
    )
    max_queued_runs: int = Field(
        default=50,
        ge=0,
        le=10000,
        description="Maximum runs that can wait in queue",
    )
    priority_weight: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Priority weight for scheduling (higher = more priority)",
    )
    queue_timeout_seconds: float = Field(
        default=DEFAULT_QUEUE_TIMEOUT_SECONDS,
        ge=10.0,
        le=86400.0,
        description="Queue timeout in seconds",
    )

    @field_validator("priority_weight")
    @classmethod
    def validate_priority_weight(cls, v: float) -> float:
        """Ensure priority weight is reasonable."""
        return round(v, 2)

    def model_dump_yaml(self) -> Dict[str, Any]:
        """Convert to YAML-friendly format."""
        return {
            "max_concurrent_runs": self.max_concurrent_runs,
            "max_concurrent_steps": self.max_concurrent_steps,
            "max_queued_runs": self.max_queued_runs,
            "priority_weight": self.priority_weight,
            "queue_timeout_seconds": self.queue_timeout_seconds,
        }


class QuotaUsage(BaseModel):
    """
    Real-time usage tracking for a namespace.

    Attributes:
        current_runs: Number of currently executing runs
        current_steps: Number of currently executing steps
        queued_runs: Number of runs waiting in queue
        active_run_ids: Set of active run IDs
        active_step_ids: Set of active step IDs
        last_updated: Last update timestamp
        total_runs_started: Total runs started (cumulative)
        total_runs_completed: Total runs completed (cumulative)
        total_queue_timeouts: Total queue timeouts (cumulative)
    """

    current_runs: int = Field(default=0, ge=0, description="Current active runs")
    current_steps: int = Field(default=0, ge=0, description="Current active steps")
    queued_runs: int = Field(default=0, ge=0, description="Runs waiting in queue")
    active_run_ids: Set[str] = Field(
        default_factory=set, description="Set of active run IDs"
    )
    active_step_ids: Set[str] = Field(
        default_factory=set, description="Set of active step IDs"
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )
    total_runs_started: int = Field(
        default=0, ge=0, description="Total runs started (cumulative)"
    )
    total_runs_completed: int = Field(
        default=0, ge=0, description="Total runs completed (cumulative)"
    )
    total_queue_timeouts: int = Field(
        default=0, ge=0, description="Total queue timeouts (cumulative)"
    )

    model_config = {"arbitrary_types_allowed": True}

    def update_timestamp(self) -> None:
        """Update the last_updated timestamp."""
        self.last_updated = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "current_runs": self.current_runs,
            "current_steps": self.current_steps,
            "queued_runs": self.queued_runs,
            "active_run_ids": list(self.active_run_ids),
            "active_step_ids": list(self.active_step_ids),
            "last_updated": self.last_updated.isoformat(),
            "total_runs_started": self.total_runs_started,
            "total_runs_completed": self.total_runs_completed,
            "total_queue_timeouts": self.total_queue_timeouts,
        }


class QuotaMetrics(BaseModel):
    """
    Metrics for quota monitoring (Prometheus-compatible).

    Attributes:
        namespace: Namespace identifier
        quota_usage_percent: Current quota utilization (0-100)
        queue_depth: Number of runs in queue
        queue_wait_time_seconds: Average wait time in queue
        runs_utilization_percent: Run slots utilization
        steps_utilization_percent: Step slots utilization
    """

    namespace: str = Field(..., description="Namespace identifier")
    quota_usage_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Overall quota utilization"
    )
    queue_depth: int = Field(default=0, ge=0, description="Queue depth")
    queue_wait_time_seconds: float = Field(
        default=0.0, ge=0.0, description="Average queue wait time"
    )
    runs_utilization_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Run slots utilization"
    )
    steps_utilization_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Step slots utilization"
    )

    def to_prometheus_format(self) -> str:
        """Convert to Prometheus exposition format."""
        lines = [
            f'greenlang_quota_usage_percent{{namespace="{self.namespace}"}} {self.quota_usage_percent}',
            f'greenlang_queue_depth{{namespace="{self.namespace}"}} {self.queue_depth}',
            f'greenlang_queue_wait_time_seconds{{namespace="{self.namespace}"}} {self.queue_wait_time_seconds}',
            f'greenlang_runs_utilization_percent{{namespace="{self.namespace}"}} {self.runs_utilization_percent}',
            f'greenlang_steps_utilization_percent{{namespace="{self.namespace}"}} {self.steps_utilization_percent}',
        ]
        return "\n".join(lines)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class QuotaEvent:
    """
    Event emitted by QuotaManager for observability.

    Attributes:
        event_type: Type of quota event
        namespace: Affected namespace
        run_id: Associated run ID (if applicable)
        step_id: Associated step ID (if applicable)
        timestamp: Event timestamp
        details: Additional event details
    """

    event_type: QuotaEventType
    namespace: str
    run_id: Optional[str] = None
    step_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "namespace": self.namespace,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass(order=True)
class QueuedRun:
    """
    A run waiting in the priority queue.

    Priority is calculated as: base_priority * namespace_weight
    Lower values = higher priority (heap queue ordering).

    Attributes:
        effective_priority: Calculated priority for ordering
        queued_at: When the run was queued
        run_id: Run identifier
        namespace: Namespace of the run
        base_priority: Original priority (1-10)
        timeout_at: When the run should timeout
    """

    effective_priority: float
    queued_at: float = field(compare=True)
    run_id: str = field(compare=False)
    namespace: str = field(compare=False)
    base_priority: int = field(compare=False)
    timeout_at: float = field(compare=False)

    def is_expired(self) -> bool:
        """Check if this queued run has expired."""
        return time.time() > self.timeout_at

    def wait_time_seconds(self) -> float:
        """Get how long this run has been waiting."""
        return time.time() - self.queued_at


# =============================================================================
# QUOTA MANAGER
# =============================================================================


class QuotaManager:
    """
    Thread-safe namespace concurrency quota manager.

    Provides:
    - Per-namespace quota configuration and enforcement
    - Admission control for runs and steps
    - Priority-weighted queue management
    - Event emission for observability
    - Comprehensive metrics

    Thread Safety:
    - All public methods are thread-safe via internal locking
    - Async methods use asyncio locks for async context

    Example:
        >>> manager = QuotaManager()
        >>> manager.set_quota("production", QuotaConfig(
        ...     max_concurrent_runs=50,
        ...     max_concurrent_steps=200,
        ...     priority_weight=2.0
        ... ))
        >>>
        >>> # Check and acquire
        >>> if manager.can_submit_run("production"):
        ...     await manager.acquire_run_slot("production", "run-123")
        ...
        >>> # Check step admission
        >>> if manager.can_start_step("production"):
        ...     await manager.acquire_step_slot("production", "run-123", "step-1")
    """

    def __init__(
        self,
        default_quota: Optional[QuotaConfig] = None,
        event_callback: Optional[Callable[[QuotaEvent], None]] = None,
    ) -> None:
        """
        Initialize QuotaManager.

        Args:
            default_quota: Default quota for namespaces without explicit config
            event_callback: Optional callback for quota events
        """
        self._default_quota = default_quota or QuotaConfig()
        self._quotas: Dict[str, QuotaConfig] = {}
        self._usage: Dict[str, QuotaUsage] = {}
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        self._event_callback = event_callback

        # Priority queue: list of QueuedRun objects (heap)
        self._queue: List[QueuedRun] = []
        self._queue_index: Dict[str, QueuedRun] = {}  # run_id -> QueuedRun

        # Wait time tracking for metrics
        self._wait_times: Dict[str, List[float]] = {}  # namespace -> recent wait times

        # Background task for queue timeout checking
        self._running = False
        self._timeout_task: Optional[asyncio.Task] = None

        logger.info("QuotaManager initialized with default quota: %s", self._default_quota)

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    async def start(self) -> None:
        """Start the quota manager background tasks."""
        if self._running:
            return

        self._running = True
        self._timeout_task = asyncio.create_task(self._timeout_check_loop())
        logger.info("QuotaManager started")

    async def stop(self) -> None:
        """Stop the quota manager gracefully."""
        self._running = False

        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass

        logger.info("QuotaManager stopped")

    async def _timeout_check_loop(self) -> None:
        """Background loop to check for queue timeouts."""
        while self._running:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds
                await self._process_queue_timeouts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in timeout check loop: %s", e, exc_info=True)

    async def _process_queue_timeouts(self) -> None:
        """Process and remove expired queue entries."""
        async with self._async_lock:
            expired_runs: List[QueuedRun] = []

            # Find expired entries
            for queued_run in list(self._queue):
                if queued_run.is_expired():
                    expired_runs.append(queued_run)

            # Remove expired entries
            for queued_run in expired_runs:
                self._remove_from_queue(queued_run.run_id)

                # Update usage
                usage = self._get_or_create_usage(queued_run.namespace)
                usage.queued_runs = max(0, usage.queued_runs - 1)
                usage.total_queue_timeouts += 1
                usage.update_timestamp()

                # Emit timeout event
                self._emit_event(
                    QuotaEvent(
                        event_type=QuotaEventType.CONCURRENCY_QUEUE_TIMEOUT,
                        namespace=queued_run.namespace,
                        run_id=queued_run.run_id,
                        details={
                            "wait_time_seconds": queued_run.wait_time_seconds(),
                            "base_priority": queued_run.base_priority,
                        },
                    )
                )

                logger.warning(
                    "Queue timeout for run %s in namespace %s (waited %.1f seconds)",
                    queued_run.run_id,
                    queued_run.namespace,
                    queued_run.wait_time_seconds(),
                )

    # =========================================================================
    # CONFIGURATION METHODS
    # =========================================================================

    def set_quota(self, namespace: str, config: QuotaConfig) -> None:
        """
        Set quota configuration for a namespace.

        Args:
            namespace: Namespace identifier
            config: Quota configuration
        """
        with self._lock:
            is_new = namespace not in self._quotas
            self._quotas[namespace] = config

            # Initialize usage if new namespace
            if namespace not in self._usage:
                self._usage[namespace] = QuotaUsage()

            logger.info(
                "Set quota for namespace '%s': runs=%d, steps=%d, weight=%.2f",
                namespace,
                config.max_concurrent_runs,
                config.max_concurrent_steps,
                config.priority_weight,
            )

            # Emit event
            event_type = (
                QuotaEventType.NAMESPACE_CREATED if is_new else QuotaEventType.QUOTA_UPDATED
            )
            self._emit_event(
                QuotaEvent(
                    event_type=event_type,
                    namespace=namespace,
                    details=config.model_dump_yaml(),
                )
            )

    def get_quota(self, namespace: str) -> QuotaConfig:
        """
        Get quota configuration for a namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            Quota configuration (default if not explicitly set)
        """
        with self._lock:
            return self._quotas.get(namespace, self._default_quota)

    def get_all_quotas(self) -> Dict[str, QuotaConfig]:
        """
        Get all namespace quotas.

        Returns:
            Dictionary of namespace -> QuotaConfig
        """
        with self._lock:
            return dict(self._quotas)

    def delete_quota(self, namespace: str) -> bool:
        """
        Delete quota configuration for a namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            True if quota was deleted
        """
        with self._lock:
            if namespace in self._quotas:
                del self._quotas[namespace]
                logger.info("Deleted quota for namespace '%s'", namespace)
                return True
            return False

    def load_from_yaml(self, path: str) -> None:
        """
        Load quota configurations from YAML file.

        Args:
            path: Path to YAML file
        """
        import yaml

        config_path = Path(path)
        if not config_path.exists():
            logger.warning("Quota config file not found: %s", path)
            return

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        if not data or "quotas" not in data:
            logger.warning("Invalid quota config format in %s", path)
            return

        quotas_data = data["quotas"]

        # Load default quota
        if "default" in quotas_data:
            self._default_quota = QuotaConfig(**quotas_data["default"])
            logger.info("Loaded default quota from %s", path)

        # Load namespace quotas
        if "namespaces" in quotas_data:
            for namespace, config in quotas_data["namespaces"].items():
                self.set_quota(namespace, QuotaConfig(**config))

        logger.info(
            "Loaded %d namespace quotas from %s",
            len(quotas_data.get("namespaces", {})),
            path,
        )

    # =========================================================================
    # USAGE TRACKING METHODS
    # =========================================================================

    def _get_or_create_usage(self, namespace: str) -> QuotaUsage:
        """Get or create usage tracking for a namespace."""
        if namespace not in self._usage:
            self._usage[namespace] = QuotaUsage()
        return self._usage[namespace]

    def get_usage(self, namespace: str) -> QuotaUsage:
        """
        Get current usage for a namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            Current usage data
        """
        with self._lock:
            return self._get_or_create_usage(namespace)

    def get_all_usage(self) -> Dict[str, QuotaUsage]:
        """
        Get usage for all namespaces.

        Returns:
            Dictionary of namespace -> QuotaUsage
        """
        with self._lock:
            return {ns: self._get_or_create_usage(ns) for ns in self._quotas}

    # =========================================================================
    # ADMISSION CONTROL METHODS
    # =========================================================================

    def can_submit_run(self, namespace: str) -> bool:
        """
        Check if a new run can be submitted (not queued).

        Args:
            namespace: Namespace identifier

        Returns:
            True if run can start immediately
        """
        with self._lock:
            quota = self.get_quota(namespace)
            usage = self._get_or_create_usage(namespace)
            return usage.current_runs < quota.max_concurrent_runs

    def can_queue_run(self, namespace: str) -> bool:
        """
        Check if a run can be added to the queue.

        Args:
            namespace: Namespace identifier

        Returns:
            True if queue has space
        """
        with self._lock:
            quota = self.get_quota(namespace)
            usage = self._get_or_create_usage(namespace)
            return usage.queued_runs < quota.max_queued_runs

    def can_start_step(self, namespace: str) -> bool:
        """
        Check if a new step can be started.

        Args:
            namespace: Namespace identifier

        Returns:
            True if step can start
        """
        with self._lock:
            quota = self.get_quota(namespace)
            usage = self._get_or_create_usage(namespace)
            return usage.current_steps < quota.max_concurrent_steps

    # =========================================================================
    # SLOT ACQUISITION METHODS (ASYNC)
    # =========================================================================

    async def acquire_run_slot(
        self,
        namespace: str,
        run_id: str,
        priority: int = 5,
        wait_for_slot: bool = True,
    ) -> bool:
        """
        Acquire a run slot for a namespace.

        Args:
            namespace: Namespace identifier
            run_id: Run identifier
            priority: Run priority (1-10, higher = more priority)
            wait_for_slot: If True, queue if no slot available

        Returns:
            True if slot acquired, False if queued or rejected
        """
        async with self._async_lock:
            quota = self.get_quota(namespace)
            usage = self._get_or_create_usage(namespace)

            # Check if slot is available
            if usage.current_runs < quota.max_concurrent_runs:
                # Acquire slot immediately
                usage.current_runs += 1
                usage.active_run_ids.add(run_id)
                usage.total_runs_started += 1
                usage.update_timestamp()

                self._emit_event(
                    QuotaEvent(
                        event_type=QuotaEventType.CONCURRENCY_SLOT_ACQUIRED,
                        namespace=namespace,
                        run_id=run_id,
                        details={"slot_type": "run", "priority": priority},
                    )
                )

                logger.debug(
                    "Run slot acquired for %s in namespace %s (%d/%d)",
                    run_id,
                    namespace,
                    usage.current_runs,
                    quota.max_concurrent_runs,
                )
                return True

            # No slot available - try to queue if allowed
            if wait_for_slot and usage.queued_runs < quota.max_queued_runs:
                self._add_to_queue(namespace, run_id, priority, quota)
                usage.queued_runs += 1
                usage.update_timestamp()

                self._emit_event(
                    QuotaEvent(
                        event_type=QuotaEventType.QUOTA_EXCEEDED,
                        namespace=namespace,
                        run_id=run_id,
                        details={
                            "queued": True,
                            "queue_position": usage.queued_runs,
                            "priority": priority,
                        },
                    )
                )

                logger.info(
                    "Run %s queued in namespace %s (queue: %d/%d)",
                    run_id,
                    namespace,
                    usage.queued_runs,
                    quota.max_queued_runs,
                )
                return False

            # Cannot queue - reject
            self._emit_event(
                QuotaEvent(
                    event_type=QuotaEventType.QUOTA_EXCEEDED,
                    namespace=namespace,
                    run_id=run_id,
                    details={"queued": False, "reason": "queue_full"},
                )
            )

            logger.warning(
                "Run %s rejected in namespace %s (quota exceeded, queue full)",
                run_id,
                namespace,
            )
            return False

    async def release_run_slot(self, namespace: str, run_id: str) -> bool:
        """
        Release a run slot.

        Args:
            namespace: Namespace identifier
            run_id: Run identifier

        Returns:
            True if slot was released
        """
        async with self._async_lock:
            usage = self._get_or_create_usage(namespace)

            if run_id not in usage.active_run_ids:
                logger.warning(
                    "Attempted to release non-existent run slot: %s in %s",
                    run_id,
                    namespace,
                )
                return False

            # Release the slot
            usage.current_runs = max(0, usage.current_runs - 1)
            usage.active_run_ids.discard(run_id)
            usage.total_runs_completed += 1
            usage.update_timestamp()

            self._emit_event(
                QuotaEvent(
                    event_type=QuotaEventType.CONCURRENCY_SLOT_RELEASED,
                    namespace=namespace,
                    run_id=run_id,
                    details={"slot_type": "run"},
                )
            )

            logger.debug(
                "Run slot released for %s in namespace %s (%d remaining)",
                run_id,
                namespace,
                usage.current_runs,
            )

            # Try to dequeue next run
            await self._try_dequeue_next()
            return True

    async def acquire_step_slot(
        self, namespace: str, run_id: str, step_id: str
    ) -> bool:
        """
        Acquire a step slot.

        Args:
            namespace: Namespace identifier
            run_id: Associated run ID
            step_id: Step identifier

        Returns:
            True if slot acquired
        """
        async with self._async_lock:
            quota = self.get_quota(namespace)
            usage = self._get_or_create_usage(namespace)

            if usage.current_steps >= quota.max_concurrent_steps:
                logger.debug(
                    "Step slot denied for %s/%s in namespace %s (at limit)",
                    run_id,
                    step_id,
                    namespace,
                )
                return False

            usage.current_steps += 1
            usage.active_step_ids.add(step_id)
            usage.update_timestamp()

            logger.debug(
                "Step slot acquired for %s/%s in namespace %s (%d/%d)",
                run_id,
                step_id,
                namespace,
                usage.current_steps,
                quota.max_concurrent_steps,
            )
            return True

    async def release_step_slot(
        self, namespace: str, run_id: str, step_id: str
    ) -> bool:
        """
        Release a step slot.

        Args:
            namespace: Namespace identifier
            run_id: Associated run ID
            step_id: Step identifier

        Returns:
            True if slot was released
        """
        async with self._async_lock:
            usage = self._get_or_create_usage(namespace)

            if step_id not in usage.active_step_ids:
                return False

            usage.current_steps = max(0, usage.current_steps - 1)
            usage.active_step_ids.discard(step_id)
            usage.update_timestamp()

            logger.debug(
                "Step slot released for %s/%s in namespace %s (%d remaining)",
                run_id,
                step_id,
                namespace,
                usage.current_steps,
            )
            return True

    # =========================================================================
    # QUEUE MANAGEMENT
    # =========================================================================

    def _add_to_queue(
        self, namespace: str, run_id: str, priority: int, quota: QuotaConfig
    ) -> None:
        """Add a run to the priority queue."""
        # Calculate effective priority (lower = higher priority)
        # Invert priority (10 - priority) so higher priority values get processed first
        # Apply weight (divide by weight so higher weight = lower effective priority = processed first)
        effective_priority = (10 - priority) / quota.priority_weight

        now = time.time()
        queued_run = QueuedRun(
            effective_priority=effective_priority,
            queued_at=now,
            run_id=run_id,
            namespace=namespace,
            base_priority=priority,
            timeout_at=now + quota.queue_timeout_seconds,
        )

        heapq.heappush(self._queue, queued_run)
        self._queue_index[run_id] = queued_run

    def _remove_from_queue(self, run_id: str) -> Optional[QueuedRun]:
        """Remove a run from the queue."""
        if run_id not in self._queue_index:
            return None

        queued_run = self._queue_index.pop(run_id)
        # Mark as removed by setting a flag (lazy removal)
        # The actual removal happens when we try to dequeue
        return queued_run

    async def _try_dequeue_next(self) -> Optional[str]:
        """
        Try to dequeue and start the next highest priority run.

        Returns:
            Run ID if a run was dequeued and started, None otherwise
        """
        while self._queue:
            # Peek at the highest priority item
            queued_run = self._queue[0]

            # Skip if already removed
            if queued_run.run_id not in self._queue_index:
                heapq.heappop(self._queue)
                continue

            # Skip if expired
            if queued_run.is_expired():
                heapq.heappop(self._queue)
                del self._queue_index[queued_run.run_id]
                continue

            # Check if we can start this run
            namespace = queued_run.namespace
            quota = self.get_quota(namespace)
            usage = self._get_or_create_usage(namespace)

            if usage.current_runs < quota.max_concurrent_runs:
                # Dequeue and start
                heapq.heappop(self._queue)
                del self._queue_index[queued_run.run_id]

                # Update usage
                usage.current_runs += 1
                usage.active_run_ids.add(queued_run.run_id)
                usage.queued_runs = max(0, usage.queued_runs - 1)
                usage.total_runs_started += 1
                usage.update_timestamp()

                # Track wait time
                wait_time = queued_run.wait_time_seconds()
                if namespace not in self._wait_times:
                    self._wait_times[namespace] = []
                self._wait_times[namespace].append(wait_time)
                # Keep only last 100 wait times
                if len(self._wait_times[namespace]) > 100:
                    self._wait_times[namespace] = self._wait_times[namespace][-100:]

                self._emit_event(
                    QuotaEvent(
                        event_type=QuotaEventType.CONCURRENCY_SLOT_ACQUIRED,
                        namespace=namespace,
                        run_id=queued_run.run_id,
                        details={
                            "slot_type": "run",
                            "dequeued": True,
                            "wait_time_seconds": wait_time,
                            "priority": queued_run.base_priority,
                        },
                    )
                )

                logger.info(
                    "Dequeued run %s in namespace %s (waited %.1f seconds)",
                    queued_run.run_id,
                    namespace,
                    wait_time,
                )
                return queued_run.run_id

            # Cannot start this run yet, stop trying
            break

        return None

    def get_queue_position(self, run_id: str) -> Optional[int]:
        """
        Get the queue position for a run.

        Args:
            run_id: Run identifier

        Returns:
            Queue position (1-based) or None if not queued
        """
        with self._lock:
            if run_id not in self._queue_index:
                return None

            # Sort queue to get position
            sorted_queue = sorted(
                [qr for qr in self._queue if qr.run_id in self._queue_index]
            )
            for i, qr in enumerate(sorted_queue):
                if qr.run_id == run_id:
                    return i + 1
            return None

    def get_queue_depth(self, namespace: Optional[str] = None) -> int:
        """
        Get current queue depth.

        Args:
            namespace: Optional namespace filter

        Returns:
            Number of runs in queue
        """
        with self._lock:
            if namespace:
                return sum(
                    1
                    for qr in self._queue
                    if qr.namespace == namespace and qr.run_id in self._queue_index
                )
            return len(self._queue_index)

    # =========================================================================
    # METRICS METHODS
    # =========================================================================

    def get_metrics(self, namespace: str) -> QuotaMetrics:
        """
        Get quota metrics for a namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            Quota metrics
        """
        with self._lock:
            quota = self.get_quota(namespace)
            usage = self._get_or_create_usage(namespace)

            # Calculate utilization percentages
            runs_util = (
                (usage.current_runs / quota.max_concurrent_runs * 100)
                if quota.max_concurrent_runs > 0
                else 0.0
            )
            steps_util = (
                (usage.current_steps / quota.max_concurrent_steps * 100)
                if quota.max_concurrent_steps > 0
                else 0.0
            )
            overall_util = max(runs_util, steps_util)

            # Calculate average wait time
            wait_times = self._wait_times.get(namespace, [])
            avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0

            return QuotaMetrics(
                namespace=namespace,
                quota_usage_percent=round(overall_util, 2),
                queue_depth=usage.queued_runs,
                queue_wait_time_seconds=round(avg_wait, 2),
                runs_utilization_percent=round(runs_util, 2),
                steps_utilization_percent=round(steps_util, 2),
            )

    def get_all_metrics(self) -> List[QuotaMetrics]:
        """
        Get metrics for all namespaces.

        Returns:
            List of quota metrics
        """
        with self._lock:
            return [self.get_metrics(ns) for ns in self._quotas]

    def get_prometheus_metrics(self) -> str:
        """
        Get all metrics in Prometheus exposition format.

        Returns:
            Prometheus-formatted metrics string
        """
        metrics_lines = [
            "# HELP greenlang_quota_usage_percent Quota utilization percentage",
            "# TYPE greenlang_quota_usage_percent gauge",
            "# HELP greenlang_queue_depth Number of runs in queue",
            "# TYPE greenlang_queue_depth gauge",
            "# HELP greenlang_queue_wait_time_seconds Average queue wait time",
            "# TYPE greenlang_queue_wait_time_seconds gauge",
            "# HELP greenlang_runs_utilization_percent Run slots utilization",
            "# TYPE greenlang_runs_utilization_percent gauge",
            "# HELP greenlang_steps_utilization_percent Step slots utilization",
            "# TYPE greenlang_steps_utilization_percent gauge",
        ]

        for metrics in self.get_all_metrics():
            metrics_lines.append(metrics.to_prometheus_format())

        return "\n".join(metrics_lines)

    # =========================================================================
    # EVENT METHODS
    # =========================================================================

    def _emit_event(self, event: QuotaEvent) -> None:
        """Emit a quota event."""
        if self._event_callback:
            try:
                self._event_callback(event)
            except Exception as e:
                logger.error("Error in event callback: %s", e)

        logger.debug("Quota event: %s", event.to_dict())

    def set_event_callback(
        self, callback: Optional[Callable[[QuotaEvent], None]]
    ) -> None:
        """
        Set the event callback.

        Args:
            callback: Function to call on events
        """
        self._event_callback = callback


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "QuotaConfig",
    "QuotaUsage",
    "QuotaManager",
    "QuotaEvent",
    "QuotaEventType",
    "QueuedRun",
    "QuotaMetrics",
    "DEFAULT_QUEUE_TIMEOUT_SECONDS",
]
