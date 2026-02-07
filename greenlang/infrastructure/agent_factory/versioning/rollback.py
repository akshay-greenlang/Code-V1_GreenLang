# -*- coding: utf-8 -*-
"""
Rollback Controller - Automated and manual rollback for agent deployments.

Monitors agent health metrics and triggers automated rollback when thresholds
are exceeded. Supports manual rollback with confirmation, cooldown periods
to prevent flapping, and full rollback history tracking. Integrates with
PushGateway for batch job metrics (OBS-001 Phase 3).

Example:
    >>> config = RollbackConfig(error_rate_threshold=0.05)
    >>> controller = RollbackController(config, metrics_provider, version_setter)
    >>> result = await controller.check_and_rollback("my-agent", "1.1.0", "1.0.0")
    >>> if result:
    ...     print(f"Rolled back: {result.trigger_reason}")

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-001 Phase 3 (PushGateway Integration)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PushGateway Integration (OBS-001 Phase 3)
# ---------------------------------------------------------------------------

try:
    from greenlang.monitoring.pushgateway import (
        BatchJobMetrics,
        get_pushgateway_client,
    )
    _PUSHGATEWAY_AVAILABLE = True
except ImportError:
    _PUSHGATEWAY_AVAILABLE = False
    BatchJobMetrics = None  # type: ignore[assignment, misc]
    get_pushgateway_client = None  # type: ignore[assignment]
    logger.debug("PushGateway SDK not available; rollback metrics disabled")


# ---------------------------------------------------------------------------
# Metrics protocol
# ---------------------------------------------------------------------------


class RollbackMetricsProvider(Protocol):
    """Interface for fetching agent health metrics for rollback decisions."""

    async def get_error_rate(self, agent_key: str, version: str) -> float:
        """Return current error rate as a fraction (0.0-1.0)."""
        ...

    async def get_p99_latency_ms(self, agent_key: str, version: str) -> float:
        """Return P99 latency in milliseconds."""
        ...

    async def get_health_check_failures(self, agent_key: str, version: str) -> int:
        """Return consecutive health check failure count."""
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RollbackConfig:
    """Configuration for automated rollback triggers.

    Attributes:
        error_rate_threshold: Max error rate before triggering rollback (default 5%).
        p99_multiplier: P99 must stay below baseline * multiplier (default 2.0x).
        health_failure_count: Consecutive health failures before rollback (default 3).
        cooldown_seconds: Minimum time between rollbacks for same agent (default 300s).
    """

    error_rate_threshold: float = 0.05
    p99_multiplier: float = 2.0
    health_failure_count: int = 3
    cooldown_seconds: int = 300


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RollbackResult:
    """Outcome of a rollback operation.

    Attributes:
        success: Whether the rollback completed successfully.
        rollback_id: Unique identifier for this rollback event.
        agent_key: Agent that was rolled back.
        from_version: Version that was rolled back from.
        to_version: Version that was rolled back to.
        trigger_reason: What triggered the rollback.
        duration_ms: Time taken to execute the rollback.
        executed_at: UTC timestamp of the rollback.
        is_manual: Whether this was a manual rollback.
    """

    success: bool
    rollback_id: str
    agent_key: str
    from_version: str
    to_version: str
    trigger_reason: str
    duration_ms: float
    executed_at: str
    is_manual: bool = False


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class RollbackController:
    """Automated and manual rollback management for agent deployments.

    Monitors metrics against configured thresholds and triggers rollback
    when any threshold is exceeded. Maintains a cooldown period to prevent
    rapid oscillation.

    Attributes:
        config: Rollback configuration.
        metrics: Health metrics provider.
        version_setter: Callback to switch active agent version.
        history: List of all rollback results.
    """

    def __init__(
        self,
        config: RollbackConfig,
        metrics: RollbackMetricsProvider,
        version_setter: Optional[Callable[[str, str], Any]] = None,
        enable_pushgateway: bool = True,
    ) -> None:
        """Initialize the rollback controller.

        Args:
            config: Rollback thresholds and cooldown.
            metrics: Provider for agent health metrics.
            version_setter: Callback(agent_key, version) to activate a version.
            enable_pushgateway: Enable PushGateway batch job metrics (OBS-001).
        """
        self.config = config
        self.metrics = metrics
        self.version_setter = version_setter
        self._history: List[RollbackResult] = []
        self._cooldowns: Dict[str, float] = {}
        self._baselines: Dict[str, float] = {}

        # PushGateway batch job metrics (OBS-001 Phase 3)
        self._enable_pushgateway = enable_pushgateway and _PUSHGATEWAY_AVAILABLE
        self._pushgateway_metrics: Optional[BatchJobMetrics] = None
        if self._enable_pushgateway:
            self._pushgateway_metrics = get_pushgateway_client("rollback-job")
            logger.info("PushGateway batch metrics enabled for RollbackController")

    @property
    def history(self) -> List[RollbackResult]:
        """Return the full rollback history."""
        return list(self._history)

    def set_baseline_p99(self, agent_key: str, baseline_ms: float) -> None:
        """Set the P99 latency baseline for an agent.

        Args:
            agent_key: Agent identifier.
            baseline_ms: Baseline P99 latency in milliseconds.
        """
        self._baselines[agent_key] = baseline_ms
        logger.info("Set P99 baseline for %s: %.1fms", agent_key, baseline_ms)

    async def check_and_rollback(
        self,
        agent_key: str,
        current_version: str,
        previous_version: str,
    ) -> Optional[RollbackResult]:
        """Check metrics and automatically rollback if thresholds are exceeded.

        Args:
            agent_key: Agent to check.
            current_version: Currently deployed version.
            previous_version: Version to rollback to if needed.

        Returns:
            RollbackResult if a rollback was triggered, None otherwise.
        """
        # Check cooldown
        if self._is_in_cooldown(agent_key):
            logger.debug("Agent %s is in cooldown, skipping rollback check", agent_key)
            return None

        # Gather metrics
        error_rate = await self.metrics.get_error_rate(agent_key, current_version)
        p99_ms = await self.metrics.get_p99_latency_ms(agent_key, current_version)
        health_failures = await self.metrics.get_health_check_failures(
            agent_key, current_version
        )

        # Check error rate
        if error_rate > self.config.error_rate_threshold:
            reason = (
                f"error_rate={error_rate:.4f} exceeds "
                f"threshold={self.config.error_rate_threshold}"
            )
            return await self._execute_rollback(
                agent_key, current_version, previous_version, reason
            )

        # Check P99 latency
        baseline = self._baselines.get(agent_key, 0.0)
        if baseline > 0:
            p99_threshold = baseline * self.config.p99_multiplier
            if p99_ms > p99_threshold:
                reason = (
                    f"p99_latency={p99_ms:.1f}ms exceeds "
                    f"threshold={p99_threshold:.1f}ms "
                    f"(baseline={baseline:.1f}ms * {self.config.p99_multiplier}x)"
                )
                return await self._execute_rollback(
                    agent_key, current_version, previous_version, reason
                )

        # Check health failures
        if health_failures >= self.config.health_failure_count:
            reason = (
                f"health_check_failures={health_failures} >= "
                f"threshold={self.config.health_failure_count}"
            )
            return await self._execute_rollback(
                agent_key, current_version, previous_version, reason
            )

        return None

    async def manual_rollback(
        self,
        agent_key: str,
        current_version: str,
        target_version: str,
        reason: str = "manual rollback",
    ) -> RollbackResult:
        """Execute a manual rollback with explicit confirmation.

        Args:
            agent_key: Agent to rollback.
            current_version: Current version.
            target_version: Version to rollback to.
            reason: Human-readable reason.

        Returns:
            RollbackResult.
        """
        return await self._execute_rollback(
            agent_key, current_version, target_version, reason, is_manual=True
        )

    def get_history_for_agent(self, agent_key: str) -> List[RollbackResult]:
        """Return rollback history for a specific agent.

        Args:
            agent_key: Agent to filter by.

        Returns:
            List of RollbackResult for the agent.
        """
        return [r for r in self._history if r.agent_key == agent_key]

    async def run_monitor_loop(
        self,
        agents: Dict[str, tuple[str, str]],
        interval_seconds: float = 30.0,
    ) -> None:
        """Run a background monitoring loop for automated rollback.

        Args:
            agents: Mapping of agent_key -> (current_version, previous_version).
            interval_seconds: Time between checks.
        """
        logger.info(
            "Starting rollback monitor loop for %d agents (interval=%.1fs)",
            len(agents),
            interval_seconds,
        )
        while True:
            for agent_key, (current, previous) in agents.items():
                try:
                    result = await self.check_and_rollback(agent_key, current, previous)
                    if result:
                        # Update the agents map to reflect the rollback
                        agents[agent_key] = (previous, current)
                except Exception as exc:
                    logger.error(
                        "Error checking rollback for %s: %s", agent_key, exc
                    )
            await asyncio.sleep(interval_seconds)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_in_cooldown(self, agent_key: str) -> bool:
        """Check if an agent is in rollback cooldown."""
        last_time = self._cooldowns.get(agent_key, 0.0)
        return (time.monotonic() - last_time) < self.config.cooldown_seconds

    async def _execute_rollback(
        self,
        agent_key: str,
        from_version: str,
        to_version: str,
        reason: str,
        is_manual: bool = False,
    ) -> RollbackResult:
        """Execute the rollback and record the result."""
        start = time.monotonic()
        rollback_id = str(uuid.uuid4())

        logger.warning(
            "Executing rollback %s for %s: %s -> %s (reason: %s)",
            rollback_id,
            agent_key,
            from_version,
            to_version,
            reason,
        )

        # Start PushGateway tracking (OBS-001 Phase 3)
        if self._pushgateway_metrics:
            self._pushgateway_metrics.set_status("running")

        success = True
        try:
            if self.version_setter:
                result = self.version_setter(agent_key, to_version)
                if asyncio.iscoroutine(result):
                    await result
        except Exception as exc:
            logger.error("Rollback execution failed: %s", exc, exc_info=True)
            success = False
            reason = f"{reason} (execution failed: {exc})"

        duration_ms = (time.monotonic() - start) * 1000
        self._cooldowns[agent_key] = time.monotonic()

        # Record in PushGateway (OBS-001 Phase 3)
        if self._pushgateway_metrics:
            self._pushgateway_metrics._record_duration(duration_ms / 1000)
            self._pushgateway_metrics.record_records(1, "rollback")
            if success:
                self._pushgateway_metrics.record_success()
            else:
                self._pushgateway_metrics.record_failure("rollback_failed")
            self._pushgateway_metrics.push()

        result = RollbackResult(
            success=success,
            rollback_id=rollback_id,
            agent_key=agent_key,
            from_version=from_version,
            to_version=to_version,
            trigger_reason=reason,
            duration_ms=duration_ms,
            executed_at=datetime.now(timezone.utc).isoformat(),
            is_manual=is_manual,
        )
        self._history.append(result)
        return result
