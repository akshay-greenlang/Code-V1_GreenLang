"""
Timeout Guard - Agent Factory Resilience (INFRA-010)

Provides configurable execution timeout enforcement for agent calls.
Supports both decorator and async context manager forms. Emits warnings
at configurable escalation thresholds before cancellation.

Classes:
    - TimeoutConfig: Configuration for timeout behaviour.
    - AgentTimeoutError: Raised when an agent exceeds its execution timeout.
    - TimeoutGuard: Core timeout enforcement implementation.

Example:
    >>> guard = TimeoutGuard("calc-agent", TimeoutConfig(timeout_s=30.0))
    >>> result = await guard.execute(agent.process, data)

    >>> async with TimeoutGuard("calc-agent").scope():
    ...     result = await long_running_operation()
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeoutConfig:
    """Configuration for timeout guard behaviour.

    Attributes:
        timeout_s: Maximum execution time in seconds before cancellation.
        warn_threshold_pct: Percentage of timeout at which a warning is emitted.
            For example, 0.8 means warn when 80% of the timeout has elapsed.
        cancel_on_timeout: Whether to cancel the task on timeout (True) or
            just raise TimeoutError.
    """

    timeout_s: float = 60.0
    warn_threshold_pct: float = 0.8
    cancel_on_timeout: bool = True


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AgentTimeoutError(asyncio.TimeoutError):
    """Raised when an agent exceeds its configured execution timeout.

    Attributes:
        agent_key: The agent that timed out.
        duration_s: Actual elapsed time in seconds.
        limit_s: The configured timeout limit.
    """

    def __init__(
        self,
        agent_key: str,
        duration_s: float,
        limit_s: float,
    ) -> None:
        self.agent_key = agent_key
        self.duration_s = duration_s
        self.limit_s = limit_s
        super().__init__(
            f"Agent '{agent_key}' timed out after {duration_s:.2f}s "
            f"(limit: {limit_s:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class TimeoutMetrics:
    """Metrics for timeout guard usage.

    Attributes:
        total_executions: Total calls guarded.
        total_timeouts: Total timeout events.
        total_warnings: Total warning threshold events.
        total_succeeded: Total calls that completed within the timeout.
    """

    total_executions: int = 0
    total_timeouts: int = 0
    total_warnings: int = 0
    total_succeeded: int = 0


# ---------------------------------------------------------------------------
# Timeout Guard
# ---------------------------------------------------------------------------


class TimeoutGuard:
    """Timeout enforcement for agent execution.

    Wraps async operations with a configurable timeout. Emits warnings
    when execution approaches the timeout threshold and raises
    AgentTimeoutError when the limit is exceeded.

    Attributes:
        agent_key: Identifier of the protected agent.
        config: Timeout configuration.
        metrics: Observable metrics.
    """

    def __init__(
        self,
        agent_key: str,
        config: Optional[TimeoutConfig] = None,
    ) -> None:
        """Initialize a timeout guard.

        Args:
            agent_key: Identifier of the agent to protect.
            config: Optional timeout configuration. Uses defaults if None.
        """
        self.agent_key = agent_key
        self.config = config or TimeoutConfig()
        self.metrics = TimeoutMetrics()
        logger.debug(
            "TimeoutGuard created for '%s' (timeout=%.1fs, warn=%.0f%%)",
            agent_key, self.config.timeout_s,
            self.config.warn_threshold_pct * 100,
        )

    # ------------------------------------------------------------------
    # Execute with timeout
    # ------------------------------------------------------------------

    async def execute(
        self,
        fn: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute an async callable with timeout enforcement.

        Args:
            fn: Async callable to execute.
            *args: Positional arguments for fn.
            **kwargs: Keyword arguments for fn.

        Returns:
            The result of fn.

        Raises:
            AgentTimeoutError: If execution exceeds the timeout.
        """
        self.metrics.total_executions += 1
        start = time.perf_counter()

        # Create a warning task that fires at the warn threshold
        warn_task = asyncio.create_task(self._warn_timer(start))

        try:
            result = await asyncio.wait_for(
                fn(*args, **kwargs),
                timeout=self.config.timeout_s,
            )
            duration = time.perf_counter() - start
            self.metrics.total_succeeded += 1
            logger.debug(
                "TimeoutGuard '%s': completed in %.2fs (limit %.1fs)",
                self.agent_key, duration, self.config.timeout_s,
            )
            return result
        except asyncio.TimeoutError:
            duration = time.perf_counter() - start
            self.metrics.total_timeouts += 1
            logger.error(
                "TimeoutGuard '%s': TIMEOUT after %.2fs (limit %.1fs)",
                self.agent_key, duration, self.config.timeout_s,
            )
            raise AgentTimeoutError(
                agent_key=self.agent_key,
                duration_s=duration,
                limit_s=self.config.timeout_s,
            )
        finally:
            warn_task.cancel()
            try:
                await warn_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def scope(self) -> AsyncIterator[None]:
        """Async context manager form of timeout enforcement.

        Yields:
            None within the guarded scope.

        Raises:
            AgentTimeoutError: If the scope exceeds the timeout.

        Example:
            >>> async with guard.scope():
            ...     await do_work()
        """
        self.metrics.total_executions += 1
        start = time.perf_counter()
        warn_task = asyncio.create_task(self._warn_timer(start))

        try:
            async with asyncio.timeout(self.config.timeout_s):
                yield
            self.metrics.total_succeeded += 1
        except asyncio.TimeoutError:
            duration = time.perf_counter() - start
            self.metrics.total_timeouts += 1
            logger.error(
                "TimeoutGuard '%s': TIMEOUT in scope after %.2fs",
                self.agent_key, duration,
            )
            raise AgentTimeoutError(
                agent_key=self.agent_key,
                duration_s=duration,
                limit_s=self.config.timeout_s,
            )
        finally:
            warn_task.cancel()
            try:
                await warn_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    @staticmethod
    def as_decorator(
        agent_key: str,
        config: Optional[TimeoutConfig] = None,
    ) -> Callable:
        """Create a decorator that wraps an async function with timeout enforcement.

        Args:
            agent_key: Agent identifier.
            config: Timeout configuration.

        Returns:
            Decorator function.

        Example:
            >>> @TimeoutGuard.as_decorator("calc-agent", TimeoutConfig(timeout_s=30))
            ... async def process(data):
            ...     return await compute(data)
        """
        guard = TimeoutGuard(agent_key, config)

        def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                return await guard.execute(fn, *args, **kwargs)
            wrapper._timeout_guard = guard  # type: ignore[attr-defined]
            return wrapper

        return decorator

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _warn_timer(self, start: float) -> None:
        """Background task that emits a warning when the threshold is reached.

        Args:
            start: The monotonic start time of the guarded operation.
        """
        warn_after = self.config.timeout_s * self.config.warn_threshold_pct
        try:
            await asyncio.sleep(warn_after)
            elapsed = time.perf_counter() - start
            self.metrics.total_warnings += 1
            logger.warning(
                "TimeoutGuard '%s': approaching timeout - %.1fs elapsed "
                "(%.0f%% of %.1fs limit)",
                self.agent_key,
                elapsed,
                (elapsed / self.config.timeout_s) * 100,
                self.config.timeout_s,
            )
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot.

        Returns:
            Dictionary with configuration and metrics.
        """
        return {
            "agent_key": self.agent_key,
            "config": {
                "timeout_s": self.config.timeout_s,
                "warn_threshold_pct": self.config.warn_threshold_pct,
            },
            "metrics": {
                "total_executions": self.metrics.total_executions,
                "total_timeouts": self.metrics.total_timeouts,
                "total_warnings": self.metrics.total_warnings,
                "total_succeeded": self.metrics.total_succeeded,
            },
        }


__all__ = [
    "AgentTimeoutError",
    "TimeoutConfig",
    "TimeoutGuard",
    "TimeoutMetrics",
]
