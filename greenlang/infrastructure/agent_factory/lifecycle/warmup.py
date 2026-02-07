# -*- coding: utf-8 -*-
"""
Warmup Strategies - Pre-flight warmup for agents before they enter RUNNING.

Provides a protocol-based warmup framework with built-in strategies for
common warm-up tasks (cache priming, connection pool initialization, model
loading) and a WarmupManager that orchestrates strategy execution per agent.

Example:
    >>> manager = WarmupManager(default_timeout=120.0)
    >>> manager.register("carbon-agent", CachePrimingWarmup(cache_keys=["ef:*"]))
    >>> manager.register("carbon-agent", ConnectionPoolWarmup(dsn="postgresql://..."))
    >>> report = await manager.warmup("carbon-agent", context={"env": "prod"})
    >>> report.success
    True

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WarmupStrategy protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class WarmupStrategy(Protocol):
    """Protocol that all warmup strategies must satisfy.

    Implementors must provide:
    - ``name``: A human-readable name for the strategy.
    - ``warmup(context)``: An async method that returns True on success.
    """

    @property
    def name(self) -> str:  # pragma: no cover
        ...

    async def warmup(self, context: Dict[str, Any]) -> bool:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class WarmupStepResult:
    """Outcome of a single warmup strategy execution.

    Attributes:
        strategy_name: Name of the strategy that was executed.
        success: Whether the strategy completed successfully.
        duration_ms: Wall-clock duration in milliseconds.
        error: Error message if the strategy failed.
    """

    strategy_name: str
    success: bool
    duration_ms: float
    error: Optional[str] = None


@dataclass
class WarmupReport:
    """Aggregated warmup report for an agent.

    Attributes:
        agent_key: Agent that was warmed up.
        success: True if ALL strategies succeeded.
        total_duration_ms: Sum of individual step durations.
        steps: Individual strategy results.
        timestamp: UTC ISO-8601 timestamp.
    """

    agent_key: str
    success: bool
    total_duration_ms: float
    steps: List[WarmupStepResult] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "agent_key": self.agent_key,
            "success": self.success,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "steps": [
                {
                    "strategy": s.strategy_name,
                    "success": s.success,
                    "duration_ms": round(s.duration_ms, 2),
                    "error": s.error,
                }
                for s in self.steps
            ],
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------

class CachePrimingWarmup:
    """Warmup strategy that primes an in-memory or Redis cache.

    Simulates cache loading by invoking an optional async loader callable.
    If no loader is provided the strategy succeeds immediately (useful for
    integration testing or placeholder wiring).

    Attributes:
        cache_keys: List of key patterns to prime.
        loader: Optional async callable that performs the actual priming.
    """

    def __init__(
        self,
        cache_keys: Optional[List[str]] = None,
        loader: Optional[Any] = None,
    ) -> None:
        self.cache_keys: List[str] = cache_keys or []
        self._loader = loader

    @property
    def name(self) -> str:
        return "cache_priming"

    async def warmup(self, context: Dict[str, Any]) -> bool:
        """Prime caches for the provided keys.

        Args:
            context: Runtime context dictionary.

        Returns:
            True if all keys were primed successfully.
        """
        logger.info(
            "CachePrimingWarmup: priming %d key patterns", len(self.cache_keys)
        )
        if self._loader is not None:
            try:
                await self._loader(self.cache_keys, context)
            except Exception:
                logger.exception("Cache priming loader failed")
                return False
        return True


class ConnectionPoolWarmup:
    """Warmup strategy that initializes a database connection pool.

    Validates that a minimum number of connections can be established
    before the agent transitions to RUNNING.

    Attributes:
        dsn: Data source name / connection string.
        min_connections: Minimum connections to establish.
        connector: Optional async callable that creates the pool.
    """

    def __init__(
        self,
        dsn: str = "",
        min_connections: int = 2,
        connector: Optional[Any] = None,
    ) -> None:
        self.dsn = dsn
        self.min_connections = min_connections
        self._connector = connector

    @property
    def name(self) -> str:
        return "connection_pool"

    async def warmup(self, context: Dict[str, Any]) -> bool:
        """Initialize the connection pool.

        Args:
            context: Runtime context dictionary.

        Returns:
            True if the pool was warmed up successfully.
        """
        logger.info(
            "ConnectionPoolWarmup: establishing min %d connections",
            self.min_connections,
        )
        if self._connector is not None:
            try:
                await self._connector(self.dsn, self.min_connections, context)
            except Exception:
                logger.exception("Connection pool warmup failed")
                return False
        return True


class ModelLoadWarmup:
    """Warmup strategy that pre-loads ML / embedding models into memory.

    Attributes:
        model_name: Identifier of the model to load.
        model_loader: Optional async callable that loads the model.
    """

    def __init__(
        self,
        model_name: str = "",
        model_loader: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self._loader = model_loader

    @property
    def name(self) -> str:
        return "model_load"

    async def warmup(self, context: Dict[str, Any]) -> bool:
        """Load the model into memory.

        Args:
            context: Runtime context dictionary.

        Returns:
            True if the model loaded successfully.
        """
        logger.info("ModelLoadWarmup: loading model '%s'", self.model_name)
        if self._loader is not None:
            try:
                await self._loader(self.model_name, context)
            except Exception:
                logger.exception("Model load warmup failed for %s", self.model_name)
                return False
        return True


# ---------------------------------------------------------------------------
# WarmupManager
# ---------------------------------------------------------------------------

class WarmupManager:
    """Orchestrates warmup strategy execution for agents.

    Provides:
    - Per-agent strategy registration.
    - Timeout-bounded execution of all strategies.
    - Aggregated WarmupReport generation.

    Example:
        >>> mgr = WarmupManager(default_timeout=120.0)
        >>> mgr.register("agent-x", CachePrimingWarmup(["ef:*"]))
        >>> report = await mgr.warmup("agent-x")
    """

    def __init__(self, default_timeout: float = 120.0) -> None:
        """Initialize the warmup manager.

        Args:
            default_timeout: Global timeout in seconds for all strategies
                per agent warmup invocation.
        """
        self._strategies: Dict[str, List[WarmupStrategy]] = {}
        self._default_timeout = default_timeout
        logger.debug(
            "WarmupManager initialized (default_timeout=%.1fs)",
            default_timeout,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, agent_key: str, strategy: WarmupStrategy) -> None:
        """Register a warmup strategy for an agent.

        Args:
            agent_key: Unique agent identifier.
            strategy: A WarmupStrategy implementation.
        """
        if agent_key not in self._strategies:
            self._strategies[agent_key] = []
        self._strategies[agent_key].append(strategy)
        logger.info(
            "Registered warmup strategy '%s' for agent %s",
            strategy.name,
            agent_key,
        )

    def deregister(self, agent_key: str) -> bool:
        """Remove all warmup strategies for an agent.

        Args:
            agent_key: Agent to deregister.

        Returns:
            True if the agent had strategies registered.
        """
        return self._strategies.pop(agent_key, None) is not None

    def get_strategies(self, agent_key: str) -> List[WarmupStrategy]:
        """Return registered strategies for an agent.

        Args:
            agent_key: Agent identifier.

        Returns:
            List of registered WarmupStrategy instances.
        """
        return list(self._strategies.get(agent_key, []))

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def warmup(
        self,
        agent_key: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> WarmupReport:
        """Execute all registered warmup strategies for an agent.

        Strategies are run **sequentially** to honour potential ordering
        dependencies (e.g. pool must exist before cache priming).

        Args:
            agent_key: Agent to warm up.
            context: Optional runtime context passed to each strategy.
            timeout: Override the default timeout (seconds).

        Returns:
            A WarmupReport summarising the outcome.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        ctx = context or {}
        strategies = self._strategies.get(agent_key, [])

        if not strategies:
            logger.warning(
                "No warmup strategies registered for agent %s", agent_key
            )
            return WarmupReport(
                agent_key=agent_key,
                success=True,
                total_duration_ms=0.0,
            )

        logger.info(
            "Starting warmup for agent %s (%d strategies, timeout=%.1fs)",
            agent_key,
            len(strategies),
            effective_timeout,
        )

        steps: List[WarmupStepResult] = []
        overall_start = time.perf_counter()
        all_ok = True

        try:
            async with asyncio.timeout(effective_timeout):
                for strategy in strategies:
                    step_result = await self._run_strategy(strategy, ctx)
                    steps.append(step_result)
                    if not step_result.success:
                        all_ok = False
                        logger.error(
                            "Warmup strategy '%s' failed for agent %s: %s",
                            strategy.name,
                            agent_key,
                            step_result.error,
                        )
                        break  # Fail fast on first failure

        except TimeoutError:
            all_ok = False
            steps.append(
                WarmupStepResult(
                    strategy_name="timeout",
                    success=False,
                    duration_ms=0.0,
                    error=f"Warmup timed out after {effective_timeout}s",
                )
            )
            logger.error(
                "Warmup for agent %s timed out after %.1fs",
                agent_key,
                effective_timeout,
            )

        total_ms = (time.perf_counter() - overall_start) * 1000

        report = WarmupReport(
            agent_key=agent_key,
            success=all_ok,
            total_duration_ms=total_ms,
            steps=steps,
        )
        logger.info(
            "Warmup for agent %s %s in %.2fms",
            agent_key,
            "succeeded" if all_ok else "FAILED",
            total_ms,
        )
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    async def _run_strategy(
        strategy: WarmupStrategy,
        context: Dict[str, Any],
    ) -> WarmupStepResult:
        """Execute a single warmup strategy and capture result."""
        start = time.perf_counter()
        error: Optional[str] = None
        success = False

        try:
            success = await strategy.warmup(context)
        except Exception as exc:
            error = str(exc)
            logger.exception("Strategy '%s' raised an exception", strategy.name)

        duration_ms = (time.perf_counter() - start) * 1000

        if not success and error is None:
            error = "Strategy returned False"

        return WarmupStepResult(
            strategy_name=strategy.name,
            success=success,
            duration_ms=duration_ms,
            error=error,
        )


__all__ = [
    "CachePrimingWarmup",
    "ConnectionPoolWarmup",
    "ModelLoadWarmup",
    "WarmupManager",
    "WarmupReport",
    "WarmupStepResult",
    "WarmupStrategy",
]
