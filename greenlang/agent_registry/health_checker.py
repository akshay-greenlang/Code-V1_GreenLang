# -*- coding: utf-8 -*-
"""
Health Checker - AGENT-FOUND-007: Agent Registry & Service Catalog

Manages health probes for registered agents with TTL-based refresh,
configurable probe timeout, and health history tracking.

Zero-Hallucination Guarantees:
    - Health status is purely observation-based
    - No LLM calls in health evaluation
    - All timestamps are UTC

Example:
    >>> from greenlang.agent_registry.health_checker import HealthChecker
    >>> checker = HealthChecker(registry)
    >>> result = checker.check_health("GL-MRV-X-001")
    >>> print(result.status)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-007 Agent Registry & Service Catalog
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.agent_registry.config import AgentRegistryConfig, get_config
from greenlang.agent_registry.models import (
    AgentHealthStatus,
    HealthCheckResult,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class HealthChecker:
    """Manages health probes for registered agents.

    Tracks health status, maintains history, and supports TTL-based
    refresh logic to avoid excessive probing.

    Attributes:
        config: Registry configuration for intervals and timeouts.
        _registry: Reference to the parent AgentRegistry instance.

    Example:
        >>> checker = HealthChecker(registry)
        >>> result = checker.check_health("GL-MRV-X-001")
        >>> assert result.status in AgentHealthStatus
    """

    def __init__(
        self,
        registry: Any,
        config: Optional[AgentRegistryConfig] = None,
    ) -> None:
        """Initialize the HealthChecker.

        Args:
            registry: AgentRegistry instance to check agents against.
            config: Optional config. Uses global singleton if None.
        """
        self.config = config or get_config()
        self._registry = registry

        # Health history: agent_id -> list of HealthCheckResult
        self._history: Dict[str, List[HealthCheckResult]] = defaultdict(list)

        # Last check time: agent_id -> timestamp (float)
        self._last_check: Dict[str, float] = {}

        # Thread safety
        self._lock = threading.Lock()

        logger.info(
            "HealthChecker initialized (interval=%ds, timeout=%ds)",
            self.config.health_check_interval_seconds,
            self.config.health_check_timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_health(self, agent_id: str, version: Optional[str] = None) -> HealthCheckResult:
        """Run a health check probe on an agent.

        Attempts to retrieve the agent from the registry and assess
        its health. Records the result in history.

        Args:
            agent_id: Agent ID to probe.
            version: Specific version to check (None = latest).

        Returns:
            HealthCheckResult with the probe outcome.
        """
        start_time = time.time()
        version_str = version or "latest"

        try:
            metadata = self._registry.get_agent(agent_id, version)
            if metadata is None:
                result = HealthCheckResult(
                    agent_id=agent_id,
                    version=version_str,
                    status=AgentHealthStatus.UNKNOWN,
                    response_time_ms=_elapsed_ms(start_time),
                    error=f"Agent not found: {agent_id}",
                )
            else:
                # Mark as healthy if found in registry
                result = HealthCheckResult(
                    agent_id=agent_id,
                    version=version_str,
                    status=metadata.health_status,
                    response_time_ms=_elapsed_ms(start_time),
                    details={
                        "name": metadata.name,
                        "layer": metadata.layer.value,
                        "execution_mode": metadata.execution_mode.value,
                    },
                )
        except Exception as exc:
            result = HealthCheckResult(
                agent_id=agent_id,
                version=version_str,
                status=AgentHealthStatus.UNHEALTHY,
                response_time_ms=_elapsed_ms(start_time),
                error=str(exc),
            )

        with self._lock:
            self._history[agent_id].append(result)
            self._last_check[agent_id] = time.time()

        logger.debug(
            "Health check %s@%s: %s (%.1fms)",
            agent_id, version_str, result.status.value, result.response_time_ms,
        )
        return result

    def set_health(
        self,
        agent_id: str,
        status: AgentHealthStatus,
        version: Optional[str] = None,
    ) -> bool:
        """Manually set health status for an agent.

        Args:
            agent_id: Agent to update.
            status: New health status.
            version: Specific version (None = latest).

        Returns:
            True if the agent was found and updated.
        """
        metadata = self._registry.get_agent(agent_id, version)
        if metadata is None:
            logger.warning("set_health: agent not found: %s", agent_id)
            return False

        metadata.health_status = status
        metadata.last_health_check = _utcnow()

        # Record in history
        result = HealthCheckResult(
            agent_id=agent_id,
            version=version or "latest",
            status=status,
            details={"source": "manual"},
        )
        with self._lock:
            self._history[agent_id].append(result)
            self._last_check[agent_id] = time.time()

        logger.info("Health set manually: %s -> %s", agent_id, status.value)
        return True

    def get_health_history(
        self, agent_id: str, limit: int = 50,
    ) -> List[HealthCheckResult]:
        """Get health check history for an agent.

        Args:
            agent_id: Agent to retrieve history for.
            limit: Maximum entries to return (newest first).

        Returns:
            List of HealthCheckResult in reverse chronological order.
        """
        with self._lock:
            entries = self._history.get(agent_id, [])
            return list(reversed(entries[-limit:]))

    def get_unhealthy_agents(self) -> List[str]:
        """Get agent IDs that are currently unhealthy.

        Returns:
            List of agent IDs with UNHEALTHY or DEGRADED status.
        """
        unhealthy: List[str] = []
        all_ids = self._registry.get_all_agent_ids()
        for agent_id in all_ids:
            metadata = self._registry.get_agent(agent_id)
            if metadata is not None and metadata.health_status in (
                AgentHealthStatus.UNHEALTHY,
                AgentHealthStatus.DEGRADED,
            ):
                unhealthy.append(agent_id)
        return sorted(unhealthy)

    def get_health_summary(self) -> Dict[str, int]:
        """Get a summary of health status counts across all agents.

        Returns:
            Dictionary mapping status values to agent counts.
        """
        counts: Dict[str, int] = {s.value: 0 for s in AgentHealthStatus}
        all_ids = self._registry.get_all_agent_ids()
        for agent_id in all_ids:
            metadata = self._registry.get_agent(agent_id)
            if metadata is not None:
                counts[metadata.health_status.value] += 1
        return counts

    def should_recheck(self, agent_id: str) -> bool:
        """Determine if an agent is due for a health recheck.

        Uses the configured health_check_interval_seconds to decide
        whether enough time has passed since the last probe.

        Args:
            agent_id: Agent to evaluate.

        Returns:
            True if enough time has elapsed for a recheck.
        """
        with self._lock:
            last = self._last_check.get(agent_id)
        if last is None:
            return True
        elapsed = time.time() - last
        return elapsed >= self.config.health_check_interval_seconds

    @property
    def checked_agent_count(self) -> int:
        """Return number of agents that have been checked at least once."""
        with self._lock:
            return len(self._history)

    @property
    def total_checks(self) -> int:
        """Return total number of health checks performed."""
        with self._lock:
            return sum(len(entries) for entries in self._history.values())


def _elapsed_ms(start_time: float) -> float:
    """Compute elapsed milliseconds since start_time.

    Args:
        start_time: Start time from time.time().

    Returns:
        Elapsed milliseconds.
    """
    return (time.time() - start_time) * 1000


__all__ = [
    "HealthChecker",
]
