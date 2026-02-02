"""
Bulkhead Pattern Implementation for GreenLang

This module provides bulkhead pattern implementations
for isolating failures and limiting concurrent access.

Features:
- Semaphore-based bulkhead
- Thread pool bulkhead
- Queue-based admission control
- Timeout support
- Metrics and observability

Example:
    >>> bulkhead = SemaphoreBulkhead(config)
    >>> async with bulkhead:
    ...     await process_request()
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BulkheadType(str, Enum):
    """Bulkhead implementation types."""
    SEMAPHORE = "semaphore"
    THREAD_POOL = "thread_pool"


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead."""
    name: str = "default"
    bulkhead_type: BulkheadType = BulkheadType.SEMAPHORE
    # Semaphore settings
    max_concurrent_calls: int = 25
    max_wait_duration_ms: int = 0  # 0 = no waiting
    # Thread pool settings
    core_thread_pool_size: int = 10
    max_thread_pool_size: int = 25
    queue_capacity: int = 100
    keep_alive_duration_ms: int = 20000


class BulkheadMetrics(BaseModel):
    """Metrics for bulkhead."""
    name: str = Field(..., description="Bulkhead name")
    available_concurrent_calls: int = Field(default=0)
    max_allowed_concurrent_calls: int = Field(default=0)
    successful_calls: int = Field(default=0)
    rejected_calls: int = Field(default=0)
    finished_waiting_calls: int = Field(default=0)
    current_thread_pool_size: Optional[int] = Field(default=None)
    queue_depth: Optional[int] = Field(default=None)


class BulkheadFullError(Exception):
    """Raised when bulkhead is full and cannot accept calls."""

    def __init__(self, name: str, max_concurrent: int):
        """Initialize error."""
        self.name = name
        self.max_concurrent = max_concurrent
        super().__init__(
            f"Bulkhead '{name}' is full. "
            f"Max concurrent calls: {max_concurrent}"
        )


class Bulkhead:
    """
    Base class for bulkhead implementations.

    Bulkheads isolate failures by limiting concurrent
    access to a resource.
    """

    async def acquire(self) -> bool:
        """Acquire a slot in the bulkhead."""
        raise NotImplementedError

    async def release(self) -> None:
        """Release a slot in the bulkhead."""
        raise NotImplementedError

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with bulkhead protection."""
        raise NotImplementedError

    def get_metrics(self) -> BulkheadMetrics:
        """Get bulkhead metrics."""
        raise NotImplementedError


class SemaphoreBulkhead(Bulkhead):
    """
    Semaphore-based bulkhead.

    Uses an asyncio semaphore to limit concurrent access.
    Simple and efficient for async workloads.

    Attributes:
        config: Bulkhead configuration
        semaphore: Underlying semaphore

    Example:
        >>> config = BulkheadConfig(
        ...     name="external-api",
        ...     max_concurrent_calls=10
        ... )
        >>> bulkhead = SemaphoreBulkhead(config)
        >>> async with bulkhead:
        ...     await call_api()
    """

    def __init__(self, config: Optional[BulkheadConfig] = None):
        """
        Initialize semaphore bulkhead.

        Args:
            config: Bulkhead configuration
        """
        self.config = config or BulkheadConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_calls)
        self._successful_calls = 0
        self._rejected_calls = 0
        self._finished_waiting = 0
        self._current_calls = 0
        self._lock = asyncio.Lock()

        logger.info(
            f"SemaphoreBulkhead '{self.config.name}' initialized: "
            f"max_concurrent={self.config.max_concurrent_calls}"
        )

    async def acquire(self) -> bool:
        """
        Acquire a slot in the bulkhead.

        Returns:
            True if slot acquired
        """
        if self.config.max_wait_duration_ms > 0:
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self.config.max_wait_duration_ms / 1000
                )
                async with self._lock:
                    self._current_calls += 1
                    self._finished_waiting += 1
                return True
            except asyncio.TimeoutError:
                async with self._lock:
                    self._rejected_calls += 1
                return False
        else:
            # Try to acquire without waiting
            if self._semaphore.locked():
                async with self._lock:
                    self._rejected_calls += 1
                return False

            await self._semaphore.acquire()
            async with self._lock:
                self._current_calls += 1
            return True

    async def release(self) -> None:
        """Release a slot in the bulkhead."""
        self._semaphore.release()
        async with self._lock:
            self._current_calls -= 1

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with bulkhead protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            BulkheadFullError: If bulkhead is full
        """
        if not await self.acquire():
            raise BulkheadFullError(
                self.config.name,
                self.config.max_concurrent_calls
            )

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            async with self._lock:
                self._successful_calls += 1

            return result

        finally:
            await self.release()

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to wrap a function with bulkhead.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)

        return wrapper

    async def __aenter__(self) -> "SemaphoreBulkhead":
        """Async context manager entry."""
        if not await self.acquire():
            raise BulkheadFullError(
                self.config.name,
                self.config.max_concurrent_calls
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.release()

    def get_metrics(self) -> BulkheadMetrics:
        """Get bulkhead metrics."""
        available = self.config.max_concurrent_calls - self._current_calls
        return BulkheadMetrics(
            name=self.config.name,
            available_concurrent_calls=available,
            max_allowed_concurrent_calls=self.config.max_concurrent_calls,
            successful_calls=self._successful_calls,
            rejected_calls=self._rejected_calls,
            finished_waiting_calls=self._finished_waiting,
        )


class ThreadPoolBulkhead(Bulkhead):
    """
    Thread pool-based bulkhead.

    Uses a thread pool with queue for isolation of
    blocking operations.

    Attributes:
        config: Bulkhead configuration
        executor: Thread pool executor

    Example:
        >>> config = BulkheadConfig(
        ...     name="db-operations",
        ...     core_thread_pool_size=5,
        ...     max_thread_pool_size=10,
        ...     queue_capacity=50
        ... )
        >>> bulkhead = ThreadPoolBulkhead(config)
        >>> result = await bulkhead.execute(blocking_operation)
    """

    def __init__(self, config: Optional[BulkheadConfig] = None):
        """
        Initialize thread pool bulkhead.

        Args:
            config: Bulkhead configuration
        """
        self.config = config or BulkheadConfig()
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_thread_pool_size,
            thread_name_prefix=f"bulkhead-{self.config.name}-"
        )
        self._queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.queue_capacity
        )
        self._successful_calls = 0
        self._rejected_calls = 0
        self._active_threads = 0
        self._lock = asyncio.Lock()

        logger.info(
            f"ThreadPoolBulkhead '{self.config.name}' initialized: "
            f"pool_size={self.config.max_thread_pool_size}, "
            f"queue_capacity={self.config.queue_capacity}"
        )

    async def acquire(self) -> bool:
        """
        Acquire a slot in the bulkhead.

        For thread pool, this checks queue capacity.
        """
        try:
            # Check if queue has capacity
            if self._queue.qsize() >= self.config.queue_capacity:
                async with self._lock:
                    self._rejected_calls += 1
                return False
            return True
        except Exception:
            return False

    async def release(self) -> None:
        """Release a slot in the bulkhead."""
        async with self._lock:
            self._active_threads = max(0, self._active_threads - 1)

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a blocking function in the thread pool.

        Args:
            func: Blocking function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            BulkheadFullError: If queue is full
        """
        if not await self.acquire():
            raise BulkheadFullError(
                self.config.name,
                self.config.queue_capacity
            )

        async with self._lock:
            self._active_threads += 1

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: func(*args, **kwargs)
            )

            async with self._lock:
                self._successful_calls += 1

            return result

        finally:
            await self.release()

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to wrap a function with bulkhead.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)

        return wrapper

    def get_metrics(self) -> BulkheadMetrics:
        """Get bulkhead metrics."""
        return BulkheadMetrics(
            name=self.config.name,
            available_concurrent_calls=self.config.max_thread_pool_size - self._active_threads,
            max_allowed_concurrent_calls=self.config.max_thread_pool_size,
            successful_calls=self._successful_calls,
            rejected_calls=self._rejected_calls,
            current_thread_pool_size=self._active_threads,
            queue_depth=self._queue.qsize(),
        )

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the thread pool.

        Args:
            wait: Wait for pending tasks to complete
        """
        self._executor.shutdown(wait=wait)
        logger.info(f"ThreadPoolBulkhead '{self.config.name}' shutdown")


class AdaptiveBulkhead(Bulkhead):
    """
    Adaptive bulkhead that adjusts limits based on metrics.

    Automatically increases or decreases concurrent call
    limit based on success/failure rates.

    Attributes:
        config: Bulkhead configuration
        current_limit: Current concurrent call limit
    """

    def __init__(
        self,
        config: Optional[BulkheadConfig] = None,
        min_limit: int = 5,
        max_limit: int = 100,
        increase_threshold: float = 0.9,  # Increase if 90% successful
        decrease_threshold: float = 0.5,  # Decrease if <50% successful
        adjustment_interval_seconds: int = 10
    ):
        """
        Initialize adaptive bulkhead.

        Args:
            config: Bulkhead configuration
            min_limit: Minimum concurrent call limit
            max_limit: Maximum concurrent call limit
            increase_threshold: Success rate to increase limit
            decrease_threshold: Success rate to decrease limit
            adjustment_interval_seconds: How often to adjust
        """
        self.config = config or BulkheadConfig()
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self.adjustment_interval = adjustment_interval_seconds

        self._current_limit = self.config.max_concurrent_calls
        self._semaphore = asyncio.Semaphore(self._current_limit)
        self._window_successes = 0
        self._window_failures = 0
        self._last_adjustment = time.monotonic()
        self._current_calls = 0
        self._lock = asyncio.Lock()

        logger.info(
            f"AdaptiveBulkhead '{self.config.name}' initialized: "
            f"initial_limit={self._current_limit}"
        )

    async def acquire(self) -> bool:
        """Acquire a slot in the bulkhead."""
        await self._maybe_adjust()

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.config.max_wait_duration_ms / 1000 if self.config.max_wait_duration_ms > 0 else None
            )
            async with self._lock:
                self._current_calls += 1
            return True
        except asyncio.TimeoutError:
            return False

    async def release(self, success: bool = True) -> None:
        """Release a slot and record outcome."""
        self._semaphore.release()
        async with self._lock:
            self._current_calls -= 1
            if success:
                self._window_successes += 1
            else:
                self._window_failures += 1

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute a function with adaptive bulkhead."""
        if not await self.acquire():
            raise BulkheadFullError(self.config.name, self._current_limit)

        success = True
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception:
            success = False
            raise
        finally:
            await self.release(success)

    async def _maybe_adjust(self) -> None:
        """Adjust limit if needed."""
        now = time.monotonic()
        if now - self._last_adjustment < self.adjustment_interval:
            return

        async with self._lock:
            total = self._window_successes + self._window_failures
            if total < 10:  # Need minimum data
                return

            success_rate = self._window_successes / total

            old_limit = self._current_limit

            if success_rate >= self.increase_threshold:
                # Increase limit
                self._current_limit = min(
                    self.max_limit,
                    int(self._current_limit * 1.2)
                )
            elif success_rate < self.decrease_threshold:
                # Decrease limit
                self._current_limit = max(
                    self.min_limit,
                    int(self._current_limit * 0.8)
                )

            if self._current_limit != old_limit:
                # Recreate semaphore with new limit
                self._semaphore = asyncio.Semaphore(
                    self._current_limit - self._current_calls
                )
                logger.info(
                    f"AdaptiveBulkhead '{self.config.name}' "
                    f"adjusted limit: {old_limit} -> {self._current_limit}"
                )

            # Reset window
            self._window_successes = 0
            self._window_failures = 0
            self._last_adjustment = now

    def get_metrics(self) -> BulkheadMetrics:
        """Get bulkhead metrics."""
        return BulkheadMetrics(
            name=self.config.name,
            available_concurrent_calls=self._current_limit - self._current_calls,
            max_allowed_concurrent_calls=self._current_limit,
            successful_calls=self._window_successes,
            rejected_calls=self._window_failures,
        )

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap a function with bulkhead."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)

        return wrapper
