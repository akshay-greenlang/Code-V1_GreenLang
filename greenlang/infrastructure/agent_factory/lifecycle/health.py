# -*- coding: utf-8 -*-
"""
Health Check Registry - Liveness, readiness, and startup probes for agents.

Provides a centralized registry for scheduling and evaluating health checks
across all managed agents.  Each agent may register independent liveness,
readiness, and startup probes.  A background scheduler periodically evaluates
probes and maintains an aggregated health status per agent and globally.

Example:
    >>> registry = HealthCheckRegistry()
    >>> registry.register(
    ...     agent_key="carbon-agent",
    ...     check=HealthCheck(name="db-ping", check_fn=ping_db, interval=10),
    ...     probe_type=ProbeType.LIVENESS,
    ... )
    >>> await registry.start()
    >>> status = registry.agent_status("carbon-agent")

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
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HealthStatus(str, Enum):
    """Aggregate health status for an agent or the system."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ProbeType(str, Enum):
    """Category of health probe aligned with Kubernetes probe model."""

    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"


# Type alias for async check functions.
HealthCheckFn = Callable[[], Coroutine[Any, Any, bool]]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class HealthCheck:
    """Definition of a single health check probe.

    Attributes:
        name: Human-readable name for the check.
        check_fn: Async callable returning True (healthy) or False.
        interval: Seconds between consecutive evaluations.
        timeout: Seconds before an individual check is considered failed.
        consecutive_failures_threshold: Number of consecutive failures
            before the check is flagged unhealthy.
    """

    name: str
    check_fn: HealthCheckFn
    interval: float = 15.0
    timeout: float = 5.0
    consecutive_failures_threshold: int = 3


@dataclass
class HealthCheckResult:
    """Outcome of a single health check evaluation.

    Attributes:
        check_name: Name of the check that was evaluated.
        probe_type: Liveness, readiness, or startup.
        healthy: Whether the check passed.
        latency_ms: Duration of the check in milliseconds.
        error: Error message if the check failed.
        timestamp: UTC ISO-8601 timestamp.
    """

    check_name: str
    probe_type: ProbeType
    healthy: bool
    latency_ms: float
    error: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class _ProbeState:
    """Internal mutable state for a registered probe."""

    check: HealthCheck
    probe_type: ProbeType
    consecutive_failures: int = 0
    last_result: Optional[HealthCheckResult] = None
    task: Optional[asyncio.Task[None]] = None


@dataclass
class AgentHealthSummary:
    """Aggregated health summary for a single agent.

    Attributes:
        agent_key: Unique agent identifier.
        status: Aggregated health status.
        checks: Per-check latest results.
        last_updated: UTC ISO-8601 timestamp.
    """

    agent_key: str
    status: HealthStatus = HealthStatus.UNKNOWN
    checks: Dict[str, HealthCheckResult] = field(default_factory=dict)
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "agent_key": self.agent_key,
            "status": self.status.value,
            "checks": {
                name: {
                    "healthy": r.healthy,
                    "latency_ms": round(r.latency_ms, 2),
                    "error": r.error,
                    "timestamp": r.timestamp,
                }
                for name, r in self.checks.items()
            },
            "last_updated": self.last_updated,
        }


# ---------------------------------------------------------------------------
# HealthCheckRegistry
# ---------------------------------------------------------------------------

class HealthCheckRegistry:
    """Central registry and scheduler for agent health probes.

    Provides:
    - Per-agent registration of liveness / readiness / startup probes.
    - Background scheduling of periodic checks with configurable intervals.
    - Aggregated health status per agent and globally.
    - Automatic transition to DEGRADED / UNHEALTHY on consecutive failures.

    Example:
        >>> registry = HealthCheckRegistry()
        >>> async def check_redis() -> bool:
        ...     return True
        >>> registry.register(
        ...     "cache-agent",
        ...     HealthCheck("redis-ping", check_redis, interval=10),
        ...     ProbeType.LIVENESS,
        ... )
        >>> await registry.start()
        >>> registry.agent_status("cache-agent")
        HealthStatus.UNKNOWN  # until first check completes
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        # agent_key -> list of probe states
        self._probes: Dict[str, List[_ProbeState]] = {}
        self._running: bool = False
        self._global_status: HealthStatus = HealthStatus.UNKNOWN
        logger.debug("HealthCheckRegistry initialized")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        agent_key: str,
        check: HealthCheck,
        probe_type: ProbeType = ProbeType.LIVENESS,
    ) -> None:
        """Register a health check for an agent.

        Args:
            agent_key: Unique identifier for the agent.
            check: Health check definition.
            probe_type: Category of the probe.
        """
        if agent_key not in self._probes:
            self._probes[agent_key] = []
        self._probes[agent_key].append(
            _ProbeState(check=check, probe_type=probe_type)
        )
        logger.info(
            "Registered %s probe '%s' for agent %s",
            probe_type.value,
            check.name,
            agent_key,
        )

    def deregister(self, agent_key: str) -> bool:
        """Remove all probes for an agent and cancel running tasks.

        Args:
            agent_key: Agent to deregister.

        Returns:
            True if the agent was found and removed.
        """
        probes = self._probes.pop(agent_key, None)
        if probes is None:
            return False
        for ps in probes:
            if ps.task and not ps.task.done():
                ps.task.cancel()
        logger.info("Deregistered all probes for agent %s", agent_key)
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background health check scheduler."""
        if self._running:
            logger.warning("HealthCheckRegistry is already running")
            return
        self._running = True
        for agent_key, probe_states in self._probes.items():
            for ps in probe_states:
                ps.task = asyncio.create_task(
                    self._schedule_probe(agent_key, ps)
                )
        logger.info(
            "HealthCheckRegistry started with %d agents",
            len(self._probes),
        )

    async def stop(self) -> None:
        """Stop all background health check tasks."""
        self._running = False
        for probe_states in self._probes.values():
            for ps in probe_states:
                if ps.task and not ps.task.done():
                    ps.task.cancel()
        # Wait briefly for cancellations
        tasks = [
            ps.task
            for pss in self._probes.values()
            for ps in pss
            if ps.task and not ps.task.done()
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("HealthCheckRegistry stopped")

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def agent_status(self, agent_key: str) -> HealthStatus:
        """Return the aggregated health status for an agent.

        Args:
            agent_key: Agent identifier.

        Returns:
            Aggregated HealthStatus.
        """
        probes = self._probes.get(agent_key)
        if not probes:
            return HealthStatus.UNKNOWN
        return self._aggregate_probes(probes)

    def agent_summary(self, agent_key: str) -> AgentHealthSummary:
        """Return a full health summary for an agent.

        Args:
            agent_key: Agent identifier.

        Returns:
            AgentHealthSummary with per-check results.
        """
        probes = self._probes.get(agent_key, [])
        checks: Dict[str, HealthCheckResult] = {}
        for ps in probes:
            if ps.last_result is not None:
                checks[ps.check.name] = ps.last_result
        return AgentHealthSummary(
            agent_key=agent_key,
            status=self.agent_status(agent_key),
            checks=checks,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    def global_status(self) -> HealthStatus:
        """Return the global aggregated health status across all agents."""
        if not self._probes:
            return HealthStatus.UNKNOWN
        statuses = [self.agent_status(k) for k in self._probes]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        return HealthStatus.UNKNOWN

    def all_summaries(self) -> Dict[str, AgentHealthSummary]:
        """Return health summaries for every registered agent."""
        return {k: self.agent_summary(k) for k in self._probes}

    # ------------------------------------------------------------------
    # Manual / on-demand evaluation
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        agent_key: str,
        probe_type: Optional[ProbeType] = None,
    ) -> List[HealthCheckResult]:
        """Run all (or filtered) health checks for an agent immediately.

        Args:
            agent_key: Agent to evaluate.
            probe_type: Optional filter by probe category.

        Returns:
            List of HealthCheckResult from the evaluation.
        """
        probes = self._probes.get(agent_key, [])
        results: List[HealthCheckResult] = []
        for ps in probes:
            if probe_type is not None and ps.probe_type != probe_type:
                continue
            result = await self._run_check(ps)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internal scheduling
    # ------------------------------------------------------------------

    async def _schedule_probe(
        self,
        agent_key: str,
        ps: _ProbeState,
    ) -> None:
        """Background loop that periodically runs a single probe."""
        while self._running:
            try:
                await self._run_check(ps)
                await asyncio.sleep(ps.check.interval)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(
                    "Unexpected error in probe '%s' for agent %s",
                    ps.check.name,
                    agent_key,
                )
                await asyncio.sleep(ps.check.interval)

    async def _run_check(self, ps: _ProbeState) -> HealthCheckResult:
        """Execute a single health check with timeout handling."""
        start = time.perf_counter()
        healthy = False
        error: Optional[str] = None

        try:
            healthy = await asyncio.wait_for(
                ps.check.check_fn(),
                timeout=ps.check.timeout,
            )
        except asyncio.TimeoutError:
            error = f"Check timed out after {ps.check.timeout}s"
        except Exception as exc:
            error = str(exc)

        latency_ms = (time.perf_counter() - start) * 1000

        if healthy:
            ps.consecutive_failures = 0
        else:
            ps.consecutive_failures += 1

        result = HealthCheckResult(
            check_name=ps.check.name,
            probe_type=ps.probe_type,
            healthy=healthy,
            latency_ms=latency_ms,
            error=error,
        )
        ps.last_result = result
        return result

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_probes(probes: List[_ProbeState]) -> HealthStatus:
        """Aggregate a list of probe states into a single HealthStatus."""
        if not probes:
            return HealthStatus.UNKNOWN

        has_result = any(ps.last_result is not None for ps in probes)
        if not has_result:
            return HealthStatus.UNKNOWN

        unhealthy_count = 0
        degraded_count = 0
        checked_count = 0

        for ps in probes:
            if ps.last_result is None:
                continue
            checked_count += 1
            if ps.consecutive_failures >= ps.check.consecutive_failures_threshold:
                unhealthy_count += 1
            elif ps.consecutive_failures > 0:
                degraded_count += 1

        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        if degraded_count > 0:
            return HealthStatus.DEGRADED
        if checked_count > 0:
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN


__all__ = [
    "AgentHealthSummary",
    "HealthCheck",
    "HealthCheckFn",
    "HealthCheckRegistry",
    "HealthCheckResult",
    "HealthStatus",
    "ProbeType",
]
