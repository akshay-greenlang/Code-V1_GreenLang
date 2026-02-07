# -*- coding: utf-8 -*-
"""
Agent Lifecycle Manager - Central controller for agent lifecycle orchestration.

Manages registration, startup, shutdown, drain, retirement, and auto-restart
of agents.  Uses distributed locking (Redis) for concurrent safety in a
multi-node deployment and emits events on every state transition through a
callback registry.

Singleton pattern with async initialization ensures a single source of truth
across the application process.

Example:
    >>> config = LifecycleManagerConfig(max_restart_attempts=5)
    >>> manager = await AgentLifecycleManager.create(config)
    >>> await manager.register_agent("carbon-agent")
    >>> await manager.start_agent("carbon-agent")
    >>> status = manager.get_status("carbon-agent")

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

from greenlang.infrastructure.agent_factory.lifecycle.states import (
    AgentState,
    AgentStateMachine,
    InvalidTransitionError,
)
from greenlang.infrastructure.agent_factory.lifecycle.health import (
    HealthCheckRegistry,
    HealthStatus,
)
from greenlang.infrastructure.agent_factory.lifecycle.warmup import WarmupManager
from greenlang.infrastructure.agent_factory.lifecycle.shutdown import (
    AgentShutdownSpec,
    GracefulShutdownCoordinator,
)

logger = logging.getLogger(__name__)

# Redis import with graceful fallback
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LifecycleManagerConfig:
    """Configuration for the AgentLifecycleManager.

    Attributes:
        health_check_interval: Seconds between background health sweeps.
        drain_timeout: Default drain timeout in seconds.
        max_restart_attempts: Maximum auto-restart attempts before giving up.
        warmup_timeout: Default warmup timeout in seconds.
        restart_backoff_base: Base seconds for exponential restart backoff.
        redis_url: Optional Redis URL for distributed locking.
        lock_ttl: Redis lock TTL in seconds.
    """

    health_check_interval: float = 15.0
    drain_timeout: float = 60.0
    max_restart_attempts: int = 5
    warmup_timeout: float = 120.0
    restart_backoff_base: float = 2.0
    redis_url: Optional[str] = None
    lock_ttl: int = 30


# Type aliases
LifecycleEventCallback = Callable[
    [str, AgentState, AgentState, str], Coroutine[Any, Any, None]
]


# ---------------------------------------------------------------------------
# Agent registry entry
# ---------------------------------------------------------------------------

@dataclass
class _AgentEntry:
    """Internal mutable state for a managed agent."""

    agent_key: str
    state_machine: AgentStateMachine
    restart_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    _monitor_task: Optional[asyncio.Task[None]] = field(
        default=None, repr=False
    )


# ---------------------------------------------------------------------------
# AgentLifecycleManager
# ---------------------------------------------------------------------------

class AgentLifecycleManager:
    """Central controller for agent lifecycle state transitions.

    Responsibilities:
    - Agent registration and de-registration.
    - Validated state transitions with distributed locking.
    - Event emission on every state change.
    - Background health monitoring with auto-restart on crash.
    - Integration with WarmupManager, HealthCheckRegistry, and
      GracefulShutdownCoordinator.

    This class follows the singleton pattern; obtain an instance via
    ``AgentLifecycleManager.create(config)``.
    """

    _instance: Optional[AgentLifecycleManager] = None

    def __init__(self, config: LifecycleManagerConfig) -> None:
        """Initialize the manager (prefer ``create()`` for async setup).

        Args:
            config: Lifecycle manager configuration.
        """
        self._config = config
        self._agents: Dict[str, _AgentEntry] = {}
        self._callbacks: List[LifecycleEventCallback] = []
        self._redis: Optional[Any] = None
        self._health_registry = HealthCheckRegistry()
        self._warmup_manager = WarmupManager(
            default_timeout=config.warmup_timeout,
        )
        self._shutdown_coordinator = GracefulShutdownCoordinator(
            drain_timeout=config.drain_timeout,
        )
        self._running = False
        self._health_task: Optional[asyncio.Task[None]] = None
        logger.info("AgentLifecycleManager constructed")

    # ------------------------------------------------------------------
    # Singleton factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        config: Optional[LifecycleManagerConfig] = None,
    ) -> AgentLifecycleManager:
        """Create (or return existing) singleton instance with async init.

        Args:
            config: Optional configuration override.

        Returns:
            Initialized AgentLifecycleManager.
        """
        if cls._instance is not None:
            return cls._instance

        effective_config = config or LifecycleManagerConfig()
        instance = cls(effective_config)
        await instance._async_init()
        cls._instance = instance
        return instance

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton for testing purposes."""
        cls._instance = None

    async def _async_init(self) -> None:
        """Perform async initialization (Redis connection, background tasks)."""
        if REDIS_AVAILABLE and self._config.redis_url:
            try:
                self._redis = aioredis.from_url(
                    self._config.redis_url,
                    decode_responses=True,
                )
                await self._redis.ping()
                logger.info(
                    "Redis connected for distributed locking: %s",
                    self._config.redis_url,
                )
            except Exception:
                logger.warning(
                    "Redis not available at %s; falling back to local locks",
                    self._config.redis_url,
                )
                self._redis = None

        self._running = True
        self._health_task = asyncio.create_task(self._health_sweep_loop())
        logger.info("AgentLifecycleManager initialized and running")

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    def on_state_change(self, callback: LifecycleEventCallback) -> None:
        """Register a callback invoked on every agent state change.

        Args:
            callback: Async callable(agent_key, old_state, new_state, reason).
        """
        self._callbacks.append(callback)

    async def _emit_event(
        self,
        agent_key: str,
        old_state: AgentState,
        new_state: AgentState,
        reason: str,
    ) -> None:
        """Dispatch event to all registered callbacks."""
        for cb in self._callbacks:
            try:
                await cb(agent_key, old_state, new_state, reason)
            except Exception:
                logger.exception("State-change callback failed for %s", agent_key)

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    async def register_agent(
        self,
        agent_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register a new agent with the lifecycle manager.

        Args:
            agent_key: Unique identifier for the agent.
            metadata: Optional metadata dict stored with the entry.

        Returns:
            True if registration succeeded; False if already registered.
        """
        if agent_key in self._agents:
            logger.warning("Agent %s is already registered", agent_key)
            return False

        entry = _AgentEntry(
            agent_key=agent_key,
            state_machine=AgentStateMachine(initial_state=AgentState.CREATED),
            metadata=metadata or {},
        )
        self._agents[agent_key] = entry
        logger.info("Agent registered: %s", agent_key)
        return True

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    async def start_agent(self, agent_key: str) -> bool:
        """Drive an agent through VALIDATING -> VALIDATED -> DEPLOYING ->
        WARMING_UP -> RUNNING.

        Args:
            agent_key: Agent to start.

        Returns:
            True if the agent reached RUNNING.
        """
        entry = self._agents.get(agent_key)
        if entry is None:
            logger.error("Cannot start unregistered agent %s", agent_key)
            return False

        steps = [
            (AgentState.VALIDATING, "validation started"),
            (AgentState.VALIDATED, "validation passed"),
            (AgentState.DEPLOYING, "deployment started"),
            (AgentState.WARMING_UP, "warmup started"),
        ]

        for target, reason in steps:
            success = await self._transition(agent_key, target, reason)
            if not success:
                return False

        # Warmup execution
        warmup_report = await self._warmup_manager.warmup(
            agent_key,
            context=entry.metadata,
            timeout=self._config.warmup_timeout,
        )
        if not warmup_report.success:
            await self._transition(
                agent_key, AgentState.FAILED, "warmup failed"
            )
            return False

        return await self._transition(
            agent_key, AgentState.RUNNING, "startup complete"
        )

    async def stop_agent(self, agent_key: str, reason: str = "stop requested") -> bool:
        """Force-stop an agent immediately.

        Args:
            agent_key: Agent to stop.
            reason: Human-readable reason.

        Returns:
            True if the transition succeeded.
        """
        return await self._transition(
            agent_key, AgentState.FORCE_STOPPED, reason
        )

    async def drain_agent(
        self,
        agent_key: str,
        reason: str = "drain requested",
    ) -> bool:
        """Transition an agent into DRAINING state.

        Args:
            agent_key: Agent to drain.
            reason: Human-readable reason.

        Returns:
            True if the transition succeeded.
        """
        return await self._transition(agent_key, AgentState.DRAINING, reason)

    async def retire_agent(self, agent_key: str) -> bool:
        """Mark a drained agent as RETIRED and remove it from the registry.

        Args:
            agent_key: Agent to retire.

        Returns:
            True if the agent was retired.
        """
        success = await self._transition(
            agent_key, AgentState.RETIRED, "retirement"
        )
        if success:
            self._health_registry.deregister(agent_key)
            self._warmup_manager.deregister(agent_key)
            # Keep entry for audit trail but mark as retired
        return success

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def get_status(self, agent_key: str) -> Optional[Dict[str, Any]]:
        """Return current status of an agent.

        Args:
            agent_key: Agent to query.

        Returns:
            Status dict or None if not registered.
        """
        entry = self._agents.get(agent_key)
        if entry is None:
            return None
        return {
            "agent_key": agent_key,
            "state": entry.state_machine.current_state.value,
            "restart_count": entry.restart_count,
            "health": self._health_registry.agent_status(agent_key).value,
            "state_machine": entry.state_machine.to_dict(),
            "metadata": entry.metadata,
        }

    def list_agents(
        self,
        state_filter: Optional[AgentState] = None,
    ) -> List[Dict[str, Any]]:
        """List all managed agents, optionally filtered by state.

        Args:
            state_filter: Only include agents in this state.

        Returns:
            List of agent status dicts.
        """
        results: List[Dict[str, Any]] = []
        for key, entry in self._agents.items():
            if (
                state_filter is not None
                and entry.state_machine.current_state != state_filter
            ):
                continue
            status = self.get_status(key)
            if status is not None:
                results.append(status)
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> LifecycleManagerConfig:
        """Return the active configuration."""
        return self._config

    @property
    def health_registry(self) -> HealthCheckRegistry:
        """Return the health check registry."""
        return self._health_registry

    @property
    def warmup_manager(self) -> WarmupManager:
        """Return the warmup manager."""
        return self._warmup_manager

    @property
    def shutdown_coordinator(self) -> GracefulShutdownCoordinator:
        """Return the shutdown coordinator."""
        return self._shutdown_coordinator

    # ------------------------------------------------------------------
    # Internal: guarded transition with distributed lock
    # ------------------------------------------------------------------

    async def _transition(
        self,
        agent_key: str,
        target: AgentState,
        reason: str,
        actor: str = "lifecycle_manager",
    ) -> bool:
        """Perform a validated, lock-guarded state transition.

        Args:
            agent_key: Agent whose state to change.
            target: Desired new state.
            reason: Human-readable reason.
            actor: Who triggered the transition.

        Returns:
            True if the transition succeeded.
        """
        entry = self._agents.get(agent_key)
        if entry is None:
            logger.error("Agent %s not in registry", agent_key)
            return False

        old_state = entry.state_machine.current_state

        lock = await self._acquire_lock(agent_key)
        try:
            entry.state_machine.transition(
                target, reason=reason, actor=actor
            )
        except InvalidTransitionError as exc:
            logger.warning("Invalid transition for %s: %s", agent_key, exc)
            return False
        finally:
            await self._release_lock(agent_key, lock)

        await self._emit_event(agent_key, old_state, target, reason)
        return True

    # ------------------------------------------------------------------
    # Distributed locking
    # ------------------------------------------------------------------

    async def _acquire_lock(self, agent_key: str) -> Optional[Any]:
        """Acquire a distributed lock for the agent key."""
        if self._redis is not None:
            lock_key = f"gl:agent_factory:lock:{agent_key}"
            try:
                lock = self._redis.lock(
                    lock_key,
                    timeout=self._config.lock_ttl,
                )
                acquired = await lock.acquire(blocking_timeout=5)
                if acquired:
                    return lock
                logger.warning(
                    "Could not acquire Redis lock for %s", agent_key
                )
            except Exception:
                logger.exception("Redis lock acquisition error for %s", agent_key)
        return None

    async def _release_lock(
        self,
        agent_key: str,
        lock: Optional[Any],
    ) -> None:
        """Release a previously acquired distributed lock."""
        if lock is not None:
            try:
                await lock.release()
            except Exception:
                logger.exception(
                    "Redis lock release error for %s", agent_key
                )

    # ------------------------------------------------------------------
    # Background health sweep and auto-restart
    # ------------------------------------------------------------------

    async def _health_sweep_loop(self) -> None:
        """Periodically evaluate agent health and auto-restart crashed agents."""
        while self._running:
            try:
                await asyncio.sleep(self._config.health_check_interval)
                await self._health_sweep()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Health sweep error")

    async def _health_sweep(self) -> None:
        """Single pass of health evaluation across all running agents."""
        for key, entry in list(self._agents.items()):
            current = entry.state_machine.current_state
            if current not in {AgentState.RUNNING, AgentState.DEGRADED}:
                continue

            health = self._health_registry.agent_status(key)

            if health == HealthStatus.UNHEALTHY and current == AgentState.RUNNING:
                await self._transition(
                    key, AgentState.DEGRADED, "health check unhealthy"
                )

            if health == HealthStatus.HEALTHY and current == AgentState.DEGRADED:
                await self._transition(
                    key, AgentState.RUNNING, "health recovered"
                )

            if health == HealthStatus.UNHEALTHY and current == AgentState.DEGRADED:
                # Agent has been unhealthy for consecutive checks; attempt restart
                await self._attempt_restart(key)

    async def _attempt_restart(self, agent_key: str) -> None:
        """Attempt to auto-restart a failed agent with exponential backoff."""
        entry = self._agents.get(agent_key)
        if entry is None:
            return

        if entry.restart_count >= self._config.max_restart_attempts:
            logger.error(
                "Max restart attempts (%d) reached for agent %s",
                self._config.max_restart_attempts,
                agent_key,
            )
            await self._transition(
                agent_key,
                AgentState.FAILED,
                f"exceeded {self._config.max_restart_attempts} restart attempts",
            )
            return

        entry.restart_count += 1
        backoff = self._config.restart_backoff_base ** entry.restart_count
        logger.info(
            "Auto-restarting agent %s (attempt %d/%d, backoff %.1fs)",
            agent_key,
            entry.restart_count,
            self._config.max_restart_attempts,
            backoff,
        )

        await self._transition(
            agent_key, AgentState.FAILED, "restarting after degraded"
        )
        await asyncio.sleep(backoff)
        await self._transition(
            agent_key, AgentState.CREATED, "auto-restart"
        )
        await self.start_agent(agent_key)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Shut down the lifecycle manager and all background tasks."""
        self._running = False
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        await self._health_registry.stop()

        if self._redis is not None:
            await self._redis.close()

        logger.info("AgentLifecycleManager closed")


__all__ = [
    "AgentLifecycleManager",
    "LifecycleEventCallback",
    "LifecycleManagerConfig",
]
