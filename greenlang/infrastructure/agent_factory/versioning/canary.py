# -*- coding: utf-8 -*-
"""
Canary Controller - Progressive canary deployment with auto-promote/rollback.

Manages staged rollout of new agent versions by gradually increasing traffic
percentage. At each step, evaluates error rate and latency metrics to decide
whether to advance, pause, or rollback. Runs a background evaluation loop.

Example:
    >>> config = CanaryConfig(steps=[5, 25, 50, 100])
    >>> controller = CanaryController(config, metrics_provider)
    >>> deployment = await controller.start("my-agent", "1.0.0", "1.1.0")
    >>> await controller.evaluate(deployment.deployment_id)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics protocol
# ---------------------------------------------------------------------------


class MetricsProvider(Protocol):
    """Interface for fetching deployment metrics."""

    async def get_error_rate(self, agent_key: str, version: str) -> float:
        """Return current error rate as a fraction (0.0-1.0)."""
        ...

    async def get_p99_latency_ms(self, agent_key: str, version: str) -> float:
        """Return P99 latency in milliseconds."""
        ...

    async def get_request_count(self, agent_key: str, version: str) -> int:
        """Return total request count since deployment start."""
        ...


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CanaryStatus(str, Enum):
    """Status of a canary deployment."""

    PENDING = "pending"
    """Deployment created but not yet started."""

    IN_PROGRESS = "in_progress"
    """Canary is actively rolling out through steps."""

    PROMOTED = "promoted"
    """New version fully promoted to 100% traffic."""

    ROLLED_BACK = "rolled_back"
    """New version was rolled back due to metrics failure."""

    PAUSED = "paused"
    """Deployment paused by operator; waiting for manual resume."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanaryConfig:
    """Configuration for canary deployment behavior.

    Attributes:
        steps: List of traffic percentages for each rollout step (e.g. [5, 25, 50, 100]).
        step_duration_seconds: Minimum time to observe at each step before advancing.
        error_rate_threshold: Max error rate before triggering rollback (fraction, e.g. 0.05).
        p99_threshold_multiplier: P99 latency must stay below baseline * multiplier.
        min_requests_per_step: Minimum request count before evaluating a step.
    """

    steps: List[int] = field(default_factory=lambda: [5, 25, 50, 100])
    step_duration_seconds: int = 300
    error_rate_threshold: float = 0.05
    p99_threshold_multiplier: float = 2.0
    min_requests_per_step: int = 50


# ---------------------------------------------------------------------------
# Deployment state
# ---------------------------------------------------------------------------


@dataclass
class CanaryDeployment:
    """State of an active canary deployment.

    Attributes:
        deployment_id: Unique deployment identifier.
        agent_key: Agent being deployed.
        old_version: Current production version.
        new_version: Version being canary-tested.
        current_step: Index into the config.steps list.
        traffic_pct: Current traffic percentage routed to new version.
        status: Current deployment status.
        started_at: UTC timestamp when the deployment started.
        step_started_at: UTC timestamp when the current step began.
        baseline_p99_ms: P99 latency baseline from the old version.
        evaluation_history: List of evaluation results.
    """

    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_key: str = ""
    old_version: str = ""
    new_version: str = ""
    current_step: int = 0
    traffic_pct: int = 0
    status: CanaryStatus = CanaryStatus.PENDING
    started_at: str = ""
    step_started_at: str = ""
    baseline_p99_ms: float = 0.0
    evaluation_history: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class CanaryController:
    """Manage canary deployments with automatic evaluation.

    Attributes:
        config: Canary deployment configuration.
        metrics: Provider for runtime metrics.
        deployments: Active deployments indexed by deployment_id.
        traffic_setter: Callback to adjust traffic routing percentages.
    """

    def __init__(
        self,
        config: CanaryConfig,
        metrics: MetricsProvider,
        traffic_setter: Optional[Callable[[str, str, int], Any]] = None,
    ) -> None:
        """Initialize the canary controller.

        Args:
            config: Canary configuration.
            metrics: Metrics provider for evaluation.
            traffic_setter: Optional callback(agent_key, version, pct) to set traffic.
        """
        self.config = config
        self.metrics = metrics
        self.traffic_setter = traffic_setter
        self._deployments: Dict[str, CanaryDeployment] = {}

    @property
    def deployments(self) -> Dict[str, CanaryDeployment]:
        """Return all tracked deployments."""
        return dict(self._deployments)

    async def start(
        self,
        agent_key: str,
        old_version: str,
        new_version: str,
    ) -> CanaryDeployment:
        """Start a new canary deployment.

        Args:
            agent_key: Agent to deploy.
            old_version: Current production version.
            new_version: New version to canary-test.

        Returns:
            The created CanaryDeployment.
        """
        # Get baseline metrics from old version
        baseline_p99 = await self.metrics.get_p99_latency_ms(agent_key, old_version)
        now = datetime.now(timezone.utc).isoformat()

        first_pct = self.config.steps[0] if self.config.steps else 0
        deployment = CanaryDeployment(
            agent_key=agent_key,
            old_version=old_version,
            new_version=new_version,
            current_step=0,
            traffic_pct=first_pct,
            status=CanaryStatus.IN_PROGRESS,
            started_at=now,
            step_started_at=now,
            baseline_p99_ms=baseline_p99,
        )

        self._deployments[deployment.deployment_id] = deployment
        await self._set_traffic(agent_key, new_version, first_pct)

        logger.info(
            "Started canary deployment %s: %s %s -> %s at %d%% traffic",
            deployment.deployment_id,
            agent_key,
            old_version,
            new_version,
            first_pct,
        )
        return deployment

    async def evaluate(self, deployment_id: str) -> CanaryDeployment:
        """Evaluate the current canary step and advance or rollback.

        Args:
            deployment_id: Deployment to evaluate.

        Returns:
            Updated CanaryDeployment.

        Raises:
            KeyError: If deployment_id is not found.
        """
        dep = self._deployments.get(deployment_id)
        if dep is None:
            raise KeyError(f"Deployment not found: {deployment_id}")

        if dep.status not in (CanaryStatus.IN_PROGRESS,):
            return dep

        # Check minimum observation time
        step_start = datetime.fromisoformat(dep.step_started_at)
        elapsed = (datetime.now(timezone.utc) - step_start).total_seconds()
        if elapsed < self.config.step_duration_seconds:
            return dep

        # Gather metrics
        error_rate = await self.metrics.get_error_rate(dep.agent_key, dep.new_version)
        p99_ms = await self.metrics.get_p99_latency_ms(dep.agent_key, dep.new_version)
        request_count = await self.metrics.get_request_count(dep.agent_key, dep.new_version)

        eval_record = {
            "step": dep.current_step,
            "traffic_pct": dep.traffic_pct,
            "error_rate": error_rate,
            "p99_ms": p99_ms,
            "request_count": request_count,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        dep.evaluation_history.append(eval_record)

        # Check minimum request count
        if request_count < self.config.min_requests_per_step:
            logger.debug(
                "Canary %s: insufficient requests (%d < %d)",
                deployment_id, request_count, self.config.min_requests_per_step,
            )
            return dep

        # Check thresholds
        p99_threshold = dep.baseline_p99_ms * self.config.p99_threshold_multiplier
        if error_rate > self.config.error_rate_threshold:
            return await self._rollback(dep, f"error_rate={error_rate:.4f} > {self.config.error_rate_threshold}")

        if p99_ms > p99_threshold and dep.baseline_p99_ms > 0:
            return await self._rollback(dep, f"p99={p99_ms:.1f}ms > {p99_threshold:.1f}ms")

        # Advance to next step
        return await self._advance(dep)

    async def pause(self, deployment_id: str) -> CanaryDeployment:
        """Pause a canary deployment.

        Args:
            deployment_id: Deployment to pause.

        Returns:
            Updated deployment.
        """
        dep = self._deployments.get(deployment_id)
        if dep is None:
            raise KeyError(f"Deployment not found: {deployment_id}")
        dep.status = CanaryStatus.PAUSED
        logger.info("Paused canary deployment %s", deployment_id)
        return dep

    async def resume(self, deployment_id: str) -> CanaryDeployment:
        """Resume a paused canary deployment.

        Args:
            deployment_id: Deployment to resume.

        Returns:
            Updated deployment.
        """
        dep = self._deployments.get(deployment_id)
        if dep is None:
            raise KeyError(f"Deployment not found: {deployment_id}")
        if dep.status != CanaryStatus.PAUSED:
            raise ValueError(f"Cannot resume deployment in status {dep.status.value}")
        dep.status = CanaryStatus.IN_PROGRESS
        dep.step_started_at = datetime.now(timezone.utc).isoformat()
        logger.info("Resumed canary deployment %s", deployment_id)
        return dep

    async def run_evaluation_loop(
        self,
        interval_seconds: float = 60.0,
    ) -> None:
        """Run background evaluation loop for all active deployments.

        This is intended to be run as an asyncio task. It continuously
        evaluates active canary deployments at the specified interval.

        Args:
            interval_seconds: Time between evaluation cycles.
        """
        logger.info("Starting canary evaluation loop (interval=%.1fs)", interval_seconds)
        while True:
            active_ids = [
                did for did, dep in self._deployments.items()
                if dep.status == CanaryStatus.IN_PROGRESS
            ]
            for did in active_ids:
                try:
                    await self.evaluate(did)
                except Exception as exc:
                    logger.error("Error evaluating canary %s: %s", did, exc)
            await asyncio.sleep(interval_seconds)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _advance(self, dep: CanaryDeployment) -> CanaryDeployment:
        """Advance to the next canary step or promote."""
        next_step = dep.current_step + 1
        if next_step >= len(self.config.steps):
            dep.status = CanaryStatus.PROMOTED
            dep.traffic_pct = 100
            await self._set_traffic(dep.agent_key, dep.new_version, 100)
            logger.info("Canary %s PROMOTED: %s -> %s", dep.deployment_id, dep.old_version, dep.new_version)
        else:
            dep.current_step = next_step
            dep.traffic_pct = self.config.steps[next_step]
            dep.step_started_at = datetime.now(timezone.utc).isoformat()
            await self._set_traffic(dep.agent_key, dep.new_version, dep.traffic_pct)
            logger.info("Canary %s advanced to step %d (%d%%)", dep.deployment_id, next_step, dep.traffic_pct)
        return dep

    async def _rollback(self, dep: CanaryDeployment, reason: str) -> CanaryDeployment:
        """Rollback the canary deployment."""
        dep.status = CanaryStatus.ROLLED_BACK
        dep.traffic_pct = 0
        await self._set_traffic(dep.agent_key, dep.new_version, 0)
        dep.evaluation_history.append({
            "action": "rollback",
            "reason": reason,
            "at": datetime.now(timezone.utc).isoformat(),
        })
        logger.warning("Canary %s ROLLED BACK: %s (reason: %s)", dep.deployment_id, dep.new_version, reason)
        return dep

    async def _set_traffic(self, agent_key: str, version: str, pct: int) -> None:
        """Set traffic percentage via the configured callback."""
        if self.traffic_setter:
            result = self.traffic_setter(agent_key, version, pct)
            if asyncio.iscoroutine(result):
                await result
