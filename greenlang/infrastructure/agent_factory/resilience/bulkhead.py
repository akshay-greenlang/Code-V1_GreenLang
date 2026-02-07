"""
Bulkhead Isolation - Agent Factory Resilience (INFRA-010)

Provides semaphore-based concurrency limiting per agent to prevent
a single misbehaving agent from consuming all available resources.
Implements the Bulkhead pattern with configurable concurrency limits,
queue depth, and queue timeout.

Classes:
    - BulkheadConfig: Configuration for bulkhead isolation.
    - BulkheadFullError: Raised when the bulkhead queue is full.
    - BulkheadIsolation: Core bulkhead implementation.

Example:
    >>> config = BulkheadConfig(max_concurrent=10, queue_size=20)
    >>> bulkhead = BulkheadIsolation("calc-agent", config)
    >>> async with bulkhead.acquire():
    ...     result = await agent.process(data)
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BulkheadConfig:
    """Configuration for bulkhead isolation.

    Attributes:
        max_concurrent: Maximum concurrent executions allowed.
        queue_size: Maximum pending requests in the wait queue.
        queue_timeout_s: Maximum seconds to wait in the queue before rejection.
    """

    max_concurrent: int = 50
    queue_size: int = 100
    queue_timeout_s: float = 30.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BulkheadFullError(Exception):
    """Raised when the bulkhead queue is full and cannot accept more requests.

    Attributes:
        agent_key: The agent whose bulkhead is full.
        active_count: Number of currently active executions.
        queue_depth: Number of requests waiting in the queue.
        max_concurrent: The configured max concurrent limit.
        max_queue: The configured max queue size.
    """

    def __init__(
        self,
        agent_key: str,
        active_count: int,
        queue_depth: int,
        max_concurrent: int,
        max_queue: int,
    ) -> None:
        self.agent_key = agent_key
        self.active_count = active_count
        self.queue_depth = queue_depth
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        super().__init__(
            f"Bulkhead for '{agent_key}' is full: "
            f"{active_count}/{max_concurrent} active, "
            f"{queue_depth}/{max_queue} queued."
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class BulkheadMetrics:
    """Observable metrics for a single bulkhead.

    Attributes:
        total_acquired: Total times the bulkhead was successfully acquired.
        total_rejected: Total times a request was rejected (queue full or timeout).
        total_timeouts: Total times a request timed out waiting in the queue.
        peak_active: Highest observed concurrent active count.
        peak_queue_depth: Highest observed queue depth.
    """

    total_acquired: int = 0
    total_rejected: int = 0
    total_timeouts: int = 0
    peak_active: int = 0
    peak_queue_depth: int = 0


# ---------------------------------------------------------------------------
# Bulkhead Isolation
# ---------------------------------------------------------------------------


class BulkheadIsolation:
    """Semaphore-based concurrency limiter per agent.

    Limits the number of concurrent executions for a given agent and
    queues additional requests up to a configured limit. Requests that
    exceed the queue capacity or timeout are rejected.

    Attributes:
        agent_key: Identifier of the protected agent.
        config: Bulkhead configuration.
        metrics: Observable metrics.
    """

    # Class-level registry: agent_key -> BulkheadIsolation
    _registry: Dict[str, BulkheadIsolation] = {}

    def __init__(
        self,
        agent_key: str,
        config: Optional[BulkheadConfig] = None,
    ) -> None:
        """Initialize a bulkhead for the given agent.

        Args:
            agent_key: Unique identifier of the agent.
            config: Optional configuration. Uses defaults if None.
        """
        self.agent_key = agent_key
        self.config = config or BulkheadConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._active_count: int = 0
        self._queue_depth: int = 0
        self._lock = asyncio.Lock()
        self.metrics = BulkheadMetrics()

        BulkheadIsolation._registry[agent_key] = self
        logger.info(
            "BulkheadIsolation created for '%s' "
            "(max_concurrent=%d, queue_size=%d, timeout=%.1fs)",
            agent_key,
            self.config.max_concurrent,
            self.config.queue_size,
            self.config.queue_timeout_s,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        """Number of currently executing calls."""
        return self._active_count

    @property
    def queue_depth(self) -> int:
        """Number of calls waiting to execute."""
        return self._queue_depth

    @property
    def available(self) -> int:
        """Number of available concurrent execution slots."""
        return max(0, self.config.max_concurrent - self._active_count)

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, agent_key: str) -> Optional[BulkheadIsolation]:
        """Retrieve a bulkhead from the registry.

        Args:
            agent_key: The agent key to look up.

        Returns:
            The BulkheadIsolation instance, or None if not registered.
        """
        return cls._registry.get(agent_key)

    @classmethod
    def get_or_create(
        cls,
        agent_key: str,
        config: Optional[BulkheadConfig] = None,
    ) -> BulkheadIsolation:
        """Get an existing bulkhead or create a new one.

        Args:
            agent_key: The agent key.
            config: Configuration for a new instance.

        Returns:
            The bulkhead for the agent.
        """
        if agent_key in cls._registry:
            return cls._registry[agent_key]
        return cls(agent_key, config)

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered bulkheads. Used for testing."""
        cls._registry.clear()

    # ------------------------------------------------------------------
    # Acquire / Release
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Async context manager to acquire a bulkhead slot.

        Waits up to queue_timeout_s for a slot. Raises BulkheadFullError
        if the queue is full, or asyncio.TimeoutError if the wait times out.

        Yields:
            None once a slot is acquired.

        Raises:
            BulkheadFullError: If the queue is full.
        """
        # Check queue capacity
        async with self._lock:
            pending = self._queue_depth
            if pending >= self.config.queue_size:
                self.metrics.total_rejected += 1
                raise BulkheadFullError(
                    agent_key=self.agent_key,
                    active_count=self._active_count,
                    queue_depth=self._queue_depth,
                    max_concurrent=self.config.max_concurrent,
                    max_queue=self.config.queue_size,
                )
            self._queue_depth += 1
            if self._queue_depth > self.metrics.peak_queue_depth:
                self.metrics.peak_queue_depth = self._queue_depth

        try:
            # Wait for a semaphore slot with timeout
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self.config.queue_timeout_s,
                )
            except asyncio.TimeoutError:
                async with self._lock:
                    self._queue_depth -= 1
                    self.metrics.total_timeouts += 1
                    self.metrics.total_rejected += 1
                logger.warning(
                    "BulkheadIsolation '%s': queue timeout after %.1fs",
                    self.agent_key, self.config.queue_timeout_s,
                )
                raise BulkheadFullError(
                    agent_key=self.agent_key,
                    active_count=self._active_count,
                    queue_depth=self._queue_depth,
                    max_concurrent=self.config.max_concurrent,
                    max_queue=self.config.queue_size,
                )
        except BulkheadFullError:
            raise
        except Exception:
            async with self._lock:
                self._queue_depth -= 1
            raise

        # Slot acquired
        async with self._lock:
            self._queue_depth -= 1
            self._active_count += 1
            if self._active_count > self.metrics.peak_active:
                self.metrics.peak_active = self._active_count
            self.metrics.total_acquired += 1

        try:
            yield
        finally:
            self._semaphore.release()
            async with self._lock:
                self._active_count -= 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot of the bulkhead state.

        Returns:
            Dictionary with current state and metrics.
        """
        return {
            "agent_key": self.agent_key,
            "active_count": self._active_count,
            "queue_depth": self._queue_depth,
            "available_slots": self.available,
            "config": {
                "max_concurrent": self.config.max_concurrent,
                "queue_size": self.config.queue_size,
                "queue_timeout_s": self.config.queue_timeout_s,
            },
            "metrics": {
                "total_acquired": self.metrics.total_acquired,
                "total_rejected": self.metrics.total_rejected,
                "total_timeouts": self.metrics.total_timeouts,
                "peak_active": self.metrics.peak_active,
                "peak_queue_depth": self.metrics.peak_queue_depth,
            },
        }


__all__ = [
    "BulkheadConfig",
    "BulkheadFullError",
    "BulkheadIsolation",
    "BulkheadMetrics",
]
