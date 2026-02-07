# -*- coding: utf-8 -*-
"""
Integration tests for Agent Lifecycle Management.

Tests the full lifecycle orchestration including state machine transitions,
health check integration, warmup execution, graceful shutdown with
in-flight tasks, event emission chains, and multi-agent coordination.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.agent_factory.lifecycle.states import (
    AgentState,
    AgentStateMachine,
    InvalidTransitionError,
)
from greenlang.infrastructure.agent_factory.lifecycle.health import (
    HealthCheck,
    HealthCheckRegistry,
    HealthStatus,
    ProbeType,
)
from greenlang.infrastructure.agent_factory.lifecycle.warmup import (
    CachePrimingWarmup,
    ConnectionPoolWarmup,
    WarmupManager,
)
from greenlang.infrastructure.agent_factory.lifecycle.shutdown import (
    AgentShutdownSpec,
    GracefulShutdownCoordinator,
)
from greenlang.infrastructure.agent_factory.lifecycle.manager import (
    AgentLifecycleManager,
    LifecycleManagerConfig,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Ensure singleton is clean before each test."""
    AgentLifecycleManager.reset_singleton()
    yield
    AgentLifecycleManager.reset_singleton()


@pytest.fixture
def config() -> LifecycleManagerConfig:
    return LifecycleManagerConfig(
        health_check_interval=60.0,
        drain_timeout=2.0,
        max_restart_attempts=3,
        warmup_timeout=5.0,
        restart_backoff_base=0.01,
        redis_url=None,
    )


@pytest.fixture
async def manager(config: LifecycleManagerConfig) -> AgentLifecycleManager:
    mgr = AgentLifecycleManager(config)
    mgr._running = True
    mgr._health_task = None
    yield mgr
    mgr._running = False
    if mgr._health_task and not mgr._health_task.done():
        mgr._health_task.cancel()


# ============================================================================
# Tests
# ============================================================================


class TestLifecycleIntegration:
    """Integration tests for the full agent lifecycle."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_lifecycle_create_to_retire(
        self, manager: AgentLifecycleManager
    ) -> None:
        """Full lifecycle: create -> validate -> deploy -> run -> drain -> retire."""
        await manager.register_agent("test-agent")
        assert await manager.start_agent("test-agent") is True

        status = manager.get_status("test-agent")
        assert status is not None
        assert status["state"] == "running"

        assert await manager.drain_agent("test-agent") is True
        assert manager.get_status("test-agent")["state"] == "draining"

        assert await manager.retire_agent("test-agent") is True
        assert manager.get_status("test-agent")["state"] == "retired"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_lifecycle_operations(
        self, manager: AgentLifecycleManager
    ) -> None:
        """Multiple agents can go through lifecycle concurrently."""
        agents = [f"agent-{i}" for i in range(5)]
        for key in agents:
            await manager.register_agent(key)

        results = await asyncio.gather(
            *[manager.start_agent(key) for key in agents]
        )
        assert all(results), "All agents should reach RUNNING"

        for key in agents:
            assert manager.get_status(key)["state"] == "running"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_event_emission_through_full_lifecycle(
        self, manager: AgentLifecycleManager
    ) -> None:
        """Events are emitted for every state transition in the lifecycle."""
        events: List[Dict[str, Any]] = []

        async def event_handler(
            agent_key: str,
            old_state: AgentState,
            new_state: AgentState,
            reason: str,
        ) -> None:
            events.append({
                "agent": agent_key,
                "from": old_state.value,
                "to": new_state.value,
                "reason": reason,
            })

        manager.on_state_change(event_handler)
        await manager.register_agent("traced-agent")
        await manager.start_agent("traced-agent")

        assert len(events) >= 5
        states_reached = [e["to"] for e in events]
        assert "validating" in states_reached
        assert "validated" in states_reached
        assert "deploying" in states_reached
        assert "warming_up" in states_reached
        assert "running" in states_reached

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_warmup_execution(
        self, manager: AgentLifecycleManager
    ) -> None:
        """Warmup strategies execute during the start flow."""
        loader = AsyncMock()
        strategy = CachePrimingWarmup(cache_keys=["ef:*"], loader=loader)
        manager.warmup_manager.register("warmup-agent", strategy)

        await manager.register_agent("warmup-agent")
        result = await manager.start_agent("warmup-agent")
        assert result is True
        loader.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_warmup_failure_prevents_running(
        self, manager: AgentLifecycleManager
    ) -> None:
        """If warmup fails, agent transitions to FAILED, not RUNNING."""
        failing_loader = AsyncMock(side_effect=RuntimeError("warmup crash"))
        strategy = CachePrimingWarmup(cache_keys=["x"], loader=failing_loader)
        manager.warmup_manager.register("fail-agent", strategy)

        await manager.register_agent("fail-agent")
        result = await manager.start_agent("fail-agent")
        assert result is False
        assert manager.get_status("fail-agent")["state"] == "failed"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_graceful_shutdown_with_inflight_tasks(
        self, manager: AgentLifecycleManager
    ) -> None:
        """Shutdown coordinator drains in-flight work."""
        drained = False

        async def drain_fn() -> bool:
            nonlocal drained
            await asyncio.sleep(0.05)
            drained = True
            return True

        stop_fn = AsyncMock()

        result = await manager.shutdown_coordinator.shutdown_agent(
            "agent-a", drain_fn, stop_fn
        )
        assert result.drained is True
        assert drained is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_lifecycle_metrics_emission(
        self, manager: AgentLifecycleManager
    ) -> None:
        """Agent status includes health and state machine information."""
        await manager.register_agent("metric-agent")
        await manager.start_agent("metric-agent")

        status = manager.get_status("metric-agent")
        assert "state_machine" in status
        assert "health" in status
        assert status["restart_count"] == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_agent_lifecycle_coordination(
        self, manager: AgentLifecycleManager
    ) -> None:
        """Multiple agents can be independently managed."""
        await manager.register_agent("agent-a")
        await manager.register_agent("agent-b")
        await manager.start_agent("agent-a")

        agents = manager.list_agents()
        assert len(agents) == 2

        running = manager.list_agents(state_filter=AgentState.RUNNING)
        assert len(running) == 1
        assert running[0]["agent_key"] == "agent-a"

        created = manager.list_agents(state_filter=AgentState.CREATED)
        assert len(created) == 1
        assert created[0]["agent_key"] == "agent-b"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_health_check_integration(
        self, manager: AgentLifecycleManager
    ) -> None:
        """Health checks can be registered and evaluated per agent."""
        async def healthy_check() -> bool:
            return True

        manager.health_registry.register(
            "health-agent",
            HealthCheck(name="ping", check_fn=healthy_check, interval=5),
            ProbeType.LIVENESS,
        )

        results = await manager.health_registry.evaluate("health-agent")
        assert len(results) == 1
        assert results[0].healthy is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_force_stop_from_running(
        self, manager: AgentLifecycleManager
    ) -> None:
        """Force stopping a running agent transitions to FORCE_STOPPED."""
        await manager.register_agent("stop-test")
        await manager.start_agent("stop-test")
        result = await manager.stop_agent("stop-test", "emergency")
        assert result is True
        assert manager.get_status("stop-test")["state"] == "force_stopped"
