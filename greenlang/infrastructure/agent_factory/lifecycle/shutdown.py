# -*- coding: utf-8 -*-
"""
Graceful Shutdown Coordinator - Orderly shutdown of managed agents.

Orchestrates the drain-and-stop sequence for one or many agents.  Agents
are drained first (stop accepting new tasks, wait for in-flight work to
finish) and then force-stopped after a configurable timeout.

Pre-shutdown and post-shutdown hooks allow callers to inject custom
teardown logic such as connection pool cleanup, metric flushing, or
deregistration from service discovery.

Example:
    >>> coordinator = GracefulShutdownCoordinator(drain_timeout=60.0)
    >>> coordinator.add_pre_shutdown_hook("flush-metrics", flush_fn)
    >>> await coordinator.shutdown_agent("carbon-agent", drain_fn, stop_fn)
    >>> await coordinator.shutdown_all(agents_map)

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
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Async callable with no arguments (or the agent key as the sole argument).
ShutdownHookFn = Callable[..., Coroutine[Any, Any, None]]
DrainFn = Callable[[], Coroutine[Any, Any, bool]]
StopFn = Callable[[], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class AgentShutdownResult:
    """Outcome of shutting down a single agent.

    Attributes:
        agent_key: Identifier of the agent.
        drained: True if the agent drained before timeout.
        forced: True if a force-stop was required.
        duration_ms: Wall-clock duration of the shutdown.
        error: Error message if something went wrong.
        timestamp: UTC ISO-8601 completion timestamp.
    """

    agent_key: str
    drained: bool = False
    forced: bool = False
    duration_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ShutdownReport:
    """Aggregated result of shutting down multiple agents.

    Attributes:
        total_agents: Number of agents targeted for shutdown.
        successful: Number of agents that shut down cleanly.
        forced: Number of agents that required force-stop.
        failed: Number of agents that encountered errors.
        total_duration_ms: Elapsed wall-clock time for the full shutdown.
        results: Per-agent shutdown results.
        timestamp: UTC ISO-8601 timestamp.
    """

    total_agents: int = 0
    successful: int = 0
    forced: int = 0
    failed: int = 0
    total_duration_ms: float = 0.0
    results: List[AgentShutdownResult] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "total_agents": self.total_agents,
            "successful": self.successful,
            "forced": self.forced,
            "failed": self.failed,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "results": [
                {
                    "agent_key": r.agent_key,
                    "drained": r.drained,
                    "forced": r.forced,
                    "duration_ms": round(r.duration_ms, 2),
                    "error": r.error,
                }
                for r in self.results
            ],
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# GracefulShutdownCoordinator
# ---------------------------------------------------------------------------

class GracefulShutdownCoordinator:
    """Coordinates graceful shutdown of managed agents.

    Lifecycle per agent:
        1. Run pre-shutdown hooks.
        2. Enter drain mode via the caller-supplied ``drain_fn``.
        3. Wait up to ``drain_timeout`` seconds for in-flight work.
        4. If drain did not complete, force-stop the agent.
        5. Run post-shutdown hooks.

    Multiple agents may be shut down concurrently with ``shutdown_all``.

    Attributes:
        drain_timeout: Default maximum seconds to wait for drain.
        force_stop_timeout: Maximum seconds to wait for a force-stop.
    """

    def __init__(
        self,
        drain_timeout: float = 60.0,
        force_stop_timeout: float = 10.0,
    ) -> None:
        """Initialize the coordinator.

        Args:
            drain_timeout: Seconds to wait for graceful drain.
            force_stop_timeout: Seconds to wait for the force-stop call.
        """
        self._drain_timeout = drain_timeout
        self._force_stop_timeout = force_stop_timeout
        self._pre_hooks: List[_HookEntry] = []
        self._post_hooks: List[_HookEntry] = []
        logger.debug(
            "GracefulShutdownCoordinator initialized "
            "(drain_timeout=%.1fs, force_stop_timeout=%.1fs)",
            drain_timeout,
            force_stop_timeout,
        )

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def add_pre_shutdown_hook(
        self,
        name: str,
        hook: ShutdownHookFn,
    ) -> None:
        """Register a hook invoked before drain begins.

        Args:
            name: Human-readable identifier for the hook.
            hook: Async callable.
        """
        self._pre_hooks.append(_HookEntry(name=name, fn=hook))
        logger.info("Pre-shutdown hook registered: %s", name)

    def add_post_shutdown_hook(
        self,
        name: str,
        hook: ShutdownHookFn,
    ) -> None:
        """Register a hook invoked after shutdown completes.

        Args:
            name: Human-readable identifier for the hook.
            hook: Async callable.
        """
        self._post_hooks.append(_HookEntry(name=name, fn=hook))
        logger.info("Post-shutdown hook registered: %s", name)

    def remove_hook(self, name: str) -> bool:
        """Remove a pre- or post-shutdown hook by name.

        Args:
            name: Name of the hook to remove.

        Returns:
            True if a hook was removed.
        """
        original_pre = len(self._pre_hooks)
        original_post = len(self._post_hooks)
        self._pre_hooks = [h for h in self._pre_hooks if h.name != name]
        self._post_hooks = [h for h in self._post_hooks if h.name != name]
        removed = (
            len(self._pre_hooks) < original_pre
            or len(self._post_hooks) < original_post
        )
        if removed:
            logger.info("Shutdown hook removed: %s", name)
        return removed

    # ------------------------------------------------------------------
    # Single-agent shutdown
    # ------------------------------------------------------------------

    async def shutdown_agent(
        self,
        agent_key: str,
        drain_fn: DrainFn,
        stop_fn: StopFn,
        *,
        drain_timeout: Optional[float] = None,
    ) -> AgentShutdownResult:
        """Gracefully shut down a single agent.

        Args:
            agent_key: Unique identifier for the agent.
            drain_fn: Async callable that initiates drain mode and returns
                True when all in-flight work is complete.
            stop_fn: Async callable that force-stops the agent.
            drain_timeout: Override the default drain timeout.

        Returns:
            AgentShutdownResult describing the outcome.
        """
        effective_timeout = (
            drain_timeout if drain_timeout is not None else self._drain_timeout
        )
        start = time.perf_counter()
        forced = False
        drained = False
        error: Optional[str] = None

        logger.info(
            "Initiating graceful shutdown for agent %s (timeout=%.1fs)",
            agent_key,
            effective_timeout,
        )

        try:
            # Pre-shutdown hooks
            await self._run_hooks(self._pre_hooks, agent_key)

            # Drain phase
            try:
                drained = await asyncio.wait_for(
                    drain_fn(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Drain timed out for agent %s after %.1fs, force-stopping",
                    agent_key,
                    effective_timeout,
                )
            except Exception as exc:
                logger.error(
                    "Drain failed for agent %s: %s", agent_key, exc
                )
                error = f"drain error: {exc}"

            # Force-stop if drain was incomplete
            if not drained:
                forced = True
                try:
                    await asyncio.wait_for(
                        stop_fn(),
                        timeout=self._force_stop_timeout,
                    )
                except asyncio.TimeoutError:
                    error = (
                        error or ""
                    ) + " force-stop timed out"
                    logger.error(
                        "Force-stop timed out for agent %s", agent_key
                    )
                except Exception as exc:
                    error = (error or "") + f" force-stop error: {exc}"
                    logger.error(
                        "Force-stop failed for agent %s: %s", agent_key, exc
                    )

            # Post-shutdown hooks
            await self._run_hooks(self._post_hooks, agent_key)

        except Exception as exc:
            error = str(exc)
            logger.exception("Shutdown error for agent %s", agent_key)

        duration_ms = (time.perf_counter() - start) * 1000
        result = AgentShutdownResult(
            agent_key=agent_key,
            drained=drained,
            forced=forced,
            duration_ms=duration_ms,
            error=error,
        )
        logger.info(
            "Shutdown complete for agent %s (drained=%s, forced=%s, %.2fms)",
            agent_key,
            drained,
            forced,
            duration_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Multi-agent shutdown
    # ------------------------------------------------------------------

    async def shutdown_all(
        self,
        agents: Dict[str, _AgentShutdownSpec],
        *,
        drain_timeout: Optional[float] = None,
    ) -> ShutdownReport:
        """Shut down multiple agents concurrently.

        Args:
            agents: Mapping of agent_key to _AgentShutdownSpec providing
                the drain and stop callables.
            drain_timeout: Override the default drain timeout for all.

        Returns:
            ShutdownReport with per-agent results.
        """
        start = time.perf_counter()
        logger.info("Initiating bulk shutdown of %d agents", len(agents))

        tasks = [
            self.shutdown_agent(
                agent_key=key,
                drain_fn=spec.drain_fn,
                stop_fn=spec.stop_fn,
                drain_timeout=drain_timeout,
            )
            for key, spec in agents.items()
        ]

        results: List[AgentShutdownResult] = await asyncio.gather(
            *tasks, return_exceptions=False
        )

        total_ms = (time.perf_counter() - start) * 1000
        report = ShutdownReport(
            total_agents=len(agents),
            successful=sum(1 for r in results if r.drained and not r.error),
            forced=sum(1 for r in results if r.forced),
            failed=sum(1 for r in results if r.error),
            total_duration_ms=total_ms,
            results=results,
        )
        logger.info(
            "Bulk shutdown complete: %d ok, %d forced, %d failed (%.2fms)",
            report.successful,
            report.forced,
            report.failed,
            total_ms,
        )
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _run_hooks(
        hooks: List[_HookEntry],
        agent_key: str,
    ) -> None:
        """Run a list of hooks sequentially with best-effort error handling."""
        for entry in hooks:
            try:
                await entry.fn(agent_key)
            except Exception:
                logger.exception(
                    "Shutdown hook '%s' raised an exception for agent %s",
                    entry.name,
                    agent_key,
                )


# ---------------------------------------------------------------------------
# Internal data models
# ---------------------------------------------------------------------------

@dataclass
class _HookEntry:
    """Internal container for a named hook."""

    name: str
    fn: ShutdownHookFn


@dataclass
class _AgentShutdownSpec:
    """Pair of drain/stop callables for multi-agent shutdown."""

    drain_fn: DrainFn
    stop_fn: StopFn


# Public alias so callers can build the dict for shutdown_all.
AgentShutdownSpec = _AgentShutdownSpec


__all__ = [
    "AgentShutdownResult",
    "AgentShutdownSpec",
    "GracefulShutdownCoordinator",
    "ShutdownReport",
]
