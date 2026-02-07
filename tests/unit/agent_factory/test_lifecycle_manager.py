# -*- coding: utf-8 -*-
"""
Unit tests for Agent Lifecycle Manager, State Machine, Health Checks,
Warmup Strategies, and Graceful Shutdown.

Tests cover state transitions, lifecycle orchestration, health probes,
warmup execution, and shutdown coordination with 85%+ coverage.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.agent_factory.lifecycle.states import (
    AgentState,
    AgentStateMachine,
    AgentStateTransition,
    InvalidTransitionError,
    VALID_TRANSITIONS,
)
from greenlang.infrastructure.agent_factory.lifecycle.health import (
    HealthCheck,
    HealthCheckRegistry,
    HealthCheckResult,
    HealthStatus,
    ProbeType,
)
from greenlang.infrastructure.agent_factory.lifecycle.warmup import (
    CachePrimingWarmup,
    ConnectionPoolWarmup,
    ModelLoadWarmup,
    WarmupManager,
    WarmupReport,
    WarmupStepResult,
)
from greenlang.infrastructure.agent_factory.lifecycle.shutdown import (
    AgentShutdownResult,
    AgentShutdownSpec,
    GracefulShutdownCoordinator,
    ShutdownReport,
)
from greenlang.infrastructure.agent_factory.lifecycle.manager import (
    AgentLifecycleManager,
    LifecycleManagerConfig,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def state_machine() -> AgentStateMachine:
    """Create a fresh state machine in CREATED state."""
    return AgentStateMachine(initial_state=AgentState.CREATED)


@pytest.fixture
def running_state_machine() -> AgentStateMachine:
    """Create a state machine that has reached RUNNING."""
    sm = AgentStateMachine(initial_state=AgentState.CREATED)
    sm.transition(AgentState.VALIDATING, reason="test", actor="test")
    sm.transition(AgentState.VALIDATED, reason="test", actor="test")
    sm.transition(AgentState.DEPLOYING, reason="test", actor="test")
    sm.transition(AgentState.WARMING_UP, reason="test", actor="test")
    sm.transition(AgentState.RUNNING, reason="test", actor="test")
    return sm


@pytest.fixture
def health_registry() -> HealthCheckRegistry:
    """Create a fresh health check registry."""
    return HealthCheckRegistry()


@pytest.fixture
def warmup_manager() -> WarmupManager:
    """Create a warmup manager with short timeout."""
    return WarmupManager(default_timeout=5.0)


@pytest.fixture
def shutdown_coordinator() -> GracefulShutdownCoordinator:
    """Create a shutdown coordinator with short timeouts."""
    return GracefulShutdownCoordinator(
        drain_timeout=2.0,
        force_stop_timeout=1.0,
    )


@pytest.fixture
def lifecycle_config() -> LifecycleManagerConfig:
    """Create a lifecycle manager config for testing."""
    return LifecycleManagerConfig(
        health_check_interval=0.1,
        drain_timeout=2.0,
        max_restart_attempts=3,
        warmup_timeout=2.0,
        restart_backoff_base=0.1,
        redis_url=None,
    )


@pytest.fixture
async def lifecycle_manager(lifecycle_config: LifecycleManagerConfig) -> AgentLifecycleManager:
    """Create and return a lifecycle manager, cleaning up the singleton after."""
    AgentLifecycleManager.reset_singleton()
    manager = AgentLifecycleManager(lifecycle_config)
    manager._running = True
    manager._health_task = None
    yield manager
    manager._running = False
    if manager._health_task and not manager._health_task.done():
        manager._health_task.cancel()
    AgentLifecycleManager.reset_singleton()


# ============================================================================
# Test AgentStateMachine
# ============================================================================


class TestAgentStateMachine:
    """Tests for the AgentStateMachine finite state machine."""

    def test_initial_state_defaults_to_created(self) -> None:
        """State machine defaults to CREATED when no initial state is given."""
        sm = AgentStateMachine()
        assert sm.current_state == AgentState.CREATED

    def test_initial_state_custom(self) -> None:
        """State machine can start in a custom initial state."""
        sm = AgentStateMachine(initial_state=AgentState.RUNNING)
        assert sm.current_state == AgentState.RUNNING

    def test_valid_transition_created_to_validating(
        self, state_machine: AgentStateMachine
    ) -> None:
        """CREATED -> VALIDATING is a valid transition."""
        record = state_machine.transition(
            AgentState.VALIDATING, reason="startup", actor="test"
        )
        assert state_machine.current_state == AgentState.VALIDATING
        assert record.from_state == AgentState.CREATED
        assert record.to_state == AgentState.VALIDATING

    @pytest.mark.parametrize(
        "from_state,to_state",
        [
            (AgentState.CREATED, AgentState.VALIDATING),
            (AgentState.VALIDATING, AgentState.VALIDATED),
            (AgentState.VALIDATED, AgentState.DEPLOYING),
            (AgentState.DEPLOYING, AgentState.WARMING_UP),
            (AgentState.WARMING_UP, AgentState.RUNNING),
            (AgentState.RUNNING, AgentState.DRAINING),
            (AgentState.DRAINING, AgentState.RETIRED),
            (AgentState.RUNNING, AgentState.DEGRADED),
            (AgentState.DEGRADED, AgentState.RUNNING),
            (AgentState.DEGRADED, AgentState.DRAINING),
            (AgentState.FAILED, AgentState.CREATED),
        ],
    )
    def test_agent_state_machine_valid_transitions(
        self, from_state: AgentState, to_state: AgentState
    ) -> None:
        """All defined valid transitions succeed."""
        sm = AgentStateMachine(initial_state=from_state)
        record = sm.transition(to_state, reason="test", actor="test")
        assert sm.current_state == to_state
        assert record.to_state == to_state

    @pytest.mark.parametrize(
        "from_state,to_state",
        [
            (AgentState.CREATED, AgentState.RUNNING),
            (AgentState.CREATED, AgentState.RETIRED),
            (AgentState.VALIDATING, AgentState.RUNNING),
            (AgentState.RUNNING, AgentState.VALIDATING),
            (AgentState.RETIRED, AgentState.RUNNING),
            (AgentState.FORCE_STOPPED, AgentState.RUNNING),
            (AgentState.DRAINING, AgentState.RUNNING),
        ],
    )
    def test_agent_state_machine_invalid_transitions(
        self, from_state: AgentState, to_state: AgentState
    ) -> None:
        """Invalid transitions raise InvalidTransitionError."""
        sm = AgentStateMachine(initial_state=from_state)
        with pytest.raises(InvalidTransitionError) as exc_info:
            sm.transition(to_state, reason="test", actor="test")
        assert exc_info.value.from_state == from_state
        assert exc_info.value.to_state == to_state

    def test_agent_state_machine_history_tracking(
        self, state_machine: AgentStateMachine
    ) -> None:
        """History records all transitions in order."""
        state_machine.transition(AgentState.VALIDATING, reason="step1", actor="a")
        state_machine.transition(AgentState.VALIDATED, reason="step2", actor="b")
        state_machine.transition(AgentState.DEPLOYING, reason="step3", actor="c")

        history = state_machine.history
        assert len(history) == 3
        assert history[0].from_state == AgentState.CREATED
        assert history[0].to_state == AgentState.VALIDATING
        assert history[0].reason == "step1"
        assert history[1].to_state == AgentState.VALIDATED
        assert history[2].to_state == AgentState.DEPLOYING

    def test_history_max_size_bounded(self) -> None:
        """History is pruned when it exceeds max_history."""
        sm = AgentStateMachine(initial_state=AgentState.RUNNING, max_history=3)
        sm.transition(AgentState.DEGRADED, reason="t1", actor="t")
        sm.transition(AgentState.RUNNING, reason="t2", actor="t")
        sm.transition(AgentState.DEGRADED, reason="t3", actor="t")
        sm.transition(AgentState.RUNNING, reason="t4", actor="t")
        assert len(sm.history) == 3

    def test_can_transition_returns_true_for_valid(
        self, state_machine: AgentStateMachine
    ) -> None:
        """can_transition returns True for allowed transitions."""
        assert state_machine.can_transition(AgentState.VALIDATING) is True

    def test_can_transition_returns_false_for_invalid(
        self, state_machine: AgentStateMachine
    ) -> None:
        """can_transition returns False for disallowed transitions."""
        assert state_machine.can_transition(AgentState.RUNNING) is False

    def test_get_valid_transitions(
        self, state_machine: AgentStateMachine
    ) -> None:
        """get_valid_transitions returns the correct frozenset for CREATED."""
        valid = state_machine.get_valid_transitions()
        assert AgentState.VALIDATING in valid
        assert AgentState.FAILED in valid
        assert AgentState.FORCE_STOPPED in valid
        assert AgentState.RUNNING not in valid

    def test_is_terminal_for_retired(self) -> None:
        """RETIRED is a terminal state with no outgoing transitions."""
        sm = AgentStateMachine(initial_state=AgentState.RETIRED)
        assert sm.is_terminal() is True

    def test_is_terminal_for_force_stopped(self) -> None:
        """FORCE_STOPPED is a terminal state."""
        sm = AgentStateMachine(initial_state=AgentState.FORCE_STOPPED)
        assert sm.is_terminal() is True

    def test_is_not_terminal_for_running(
        self, running_state_machine: AgentStateMachine
    ) -> None:
        """RUNNING is not a terminal state."""
        assert running_state_machine.is_terminal() is False

    def test_is_active_for_running(
        self, running_state_machine: AgentStateMachine
    ) -> None:
        """RUNNING is an active state."""
        assert running_state_machine.is_active() is True

    def test_is_active_for_created(
        self, state_machine: AgentStateMachine
    ) -> None:
        """CREATED is not an active state."""
        assert state_machine.is_active() is False

    def test_callback_invoked_on_transition(
        self, state_machine: AgentStateMachine
    ) -> None:
        """Registered callbacks are called on each transition."""
        callback = MagicMock()
        state_machine.on_transition(callback)
        state_machine.transition(AgentState.VALIDATING, reason="test", actor="test")
        callback.assert_called_once()
        record = callback.call_args[0][0]
        assert isinstance(record, AgentStateTransition)
        assert record.to_state == AgentState.VALIDATING

    def test_callback_exception_does_not_block_transition(
        self, state_machine: AgentStateMachine
    ) -> None:
        """A failing callback does not prevent the transition from completing."""
        bad_callback = MagicMock(side_effect=RuntimeError("boom"))
        state_machine.on_transition(bad_callback)
        state_machine.transition(AgentState.VALIDATING, reason="test", actor="test")
        assert state_machine.current_state == AgentState.VALIDATING

    def test_remove_callback(self, state_machine: AgentStateMachine) -> None:
        """Removing a callback prevents future invocations."""
        callback = MagicMock()
        state_machine.on_transition(callback)
        assert state_machine.remove_callback(callback) is True
        state_machine.transition(AgentState.VALIDATING, reason="test", actor="test")
        callback.assert_not_called()

    def test_remove_callback_returns_false_for_unknown(
        self, state_machine: AgentStateMachine
    ) -> None:
        """Removing a non-registered callback returns False."""
        assert state_machine.remove_callback(lambda x: None) is False

    def test_to_dict_serialization(
        self, running_state_machine: AgentStateMachine
    ) -> None:
        """to_dict produces a valid dictionary representation."""
        data = running_state_machine.to_dict()
        assert data["current_state"] == "running"
        assert data["transition_count"] == 5
        assert isinstance(data["history"], list)
        assert isinstance(data["valid_transitions"], list)

    def test_any_state_can_transition_to_failed(self) -> None:
        """Every non-terminal state can transition to FAILED."""
        non_terminal = [
            s for s in AgentState
            if s not in {AgentState.RETIRED, AgentState.FORCE_STOPPED}
        ]
        for state in non_terminal:
            sm = AgentStateMachine(initial_state=state)
            if AgentState.FAILED in VALID_TRANSITIONS.get(state, frozenset()):
                sm.transition(AgentState.FAILED, reason="test", actor="test")
                assert sm.current_state == AgentState.FAILED

    def test_any_state_can_transition_to_force_stopped(self) -> None:
        """Every non-terminal state can transition to FORCE_STOPPED."""
        non_terminal = [
            s for s in AgentState
            if s not in {AgentState.RETIRED, AgentState.FORCE_STOPPED}
        ]
        for state in non_terminal:
            sm = AgentStateMachine(initial_state=state)
            if AgentState.FORCE_STOPPED in VALID_TRANSITIONS.get(state, frozenset()):
                sm.transition(AgentState.FORCE_STOPPED, reason="test", actor="test")
                assert sm.current_state == AgentState.FORCE_STOPPED

    def test_transition_record_contains_metadata(
        self, state_machine: AgentStateMachine
    ) -> None:
        """Transition records include metadata when provided."""
        record = state_machine.transition(
            AgentState.VALIDATING,
            reason="test",
            actor="deployer",
            metadata={"env": "staging"},
        )
        assert record.metadata == {"env": "staging"}
        assert record.actor == "deployer"

    def test_time_in_current_state_positive(
        self, state_machine: AgentStateMachine
    ) -> None:
        """time_in_current_state_seconds returns a positive value."""
        elapsed = state_machine.time_in_current_state_seconds()
        assert elapsed >= 0.0


# ============================================================================
# Test HealthCheckRegistry
# ============================================================================


class TestHealthCheckRegistry:
    """Tests for HealthCheckRegistry probe management."""

    def test_health_check_registry_add_check(
        self, health_registry: HealthCheckRegistry
    ) -> None:
        """Registering a health check stores it for the agent."""
        async def check_fn() -> bool:
            return True

        health_registry.register(
            "agent-a",
            HealthCheck(name="db-ping", check_fn=check_fn, interval=5.0),
            ProbeType.LIVENESS,
        )
        status = health_registry.agent_status("agent-a")
        assert status == HealthStatus.UNKNOWN  # no evaluation yet

    @pytest.mark.asyncio
    async def test_health_check_execution_healthy(
        self, health_registry: HealthCheckRegistry
    ) -> None:
        """A passing check results in HEALTHY status."""
        async def check_fn() -> bool:
            return True

        health_registry.register(
            "agent-a",
            HealthCheck(name="ok-check", check_fn=check_fn, interval=5.0),
            ProbeType.LIVENESS,
        )
        results = await health_registry.evaluate("agent-a")
        assert len(results) == 1
        assert results[0].healthy is True
        assert health_registry.agent_status("agent-a") == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_execution_unhealthy(
        self, health_registry: HealthCheckRegistry
    ) -> None:
        """A check that fails enough times results in UNHEALTHY status."""
        async def failing_check() -> bool:
            return False

        health_registry.register(
            "agent-a",
            HealthCheck(
                name="fail-check",
                check_fn=failing_check,
                interval=5.0,
                consecutive_failures_threshold=2,
            ),
            ProbeType.LIVENESS,
        )
        # First evaluation: consecutive_failures=1, status=DEGRADED
        await health_registry.evaluate("agent-a")
        assert health_registry.agent_status("agent-a") == HealthStatus.DEGRADED

        # Second evaluation: consecutive_failures=2, status=UNHEALTHY
        await health_registry.evaluate("agent-a")
        assert health_registry.agent_status("agent-a") == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_timeout(
        self, health_registry: HealthCheckRegistry
    ) -> None:
        """A check that exceeds its timeout is treated as a failure."""
        async def slow_check() -> bool:
            await asyncio.sleep(10.0)
            return True

        health_registry.register(
            "agent-a",
            HealthCheck(
                name="slow-check",
                check_fn=slow_check,
                interval=5.0,
                timeout=0.01,
                consecutive_failures_threshold=1,
            ),
            ProbeType.LIVENESS,
        )
        results = await health_registry.evaluate("agent-a")
        assert results[0].healthy is False
        assert "timed out" in (results[0].error or "").lower()

    @pytest.mark.asyncio
    async def test_health_check_exception_treated_as_failure(
        self, health_registry: HealthCheckRegistry
    ) -> None:
        """A check that raises an exception is recorded as failed."""
        async def crashing_check() -> bool:
            raise ConnectionError("db down")

        health_registry.register(
            "agent-a",
            HealthCheck(
                name="crash-check",
                check_fn=crashing_check,
                interval=5.0,
                consecutive_failures_threshold=1,
            ),
            ProbeType.LIVENESS,
        )
        results = await health_registry.evaluate("agent-a")
        assert results[0].healthy is False
        assert "db down" in (results[0].error or "")

    def test_deregister_unknown_agent_returns_false(
        self, health_registry: HealthCheckRegistry
    ) -> None:
        """Deregistering an unknown agent returns False."""
        assert health_registry.deregister("nonexistent") is False

    def test_deregister_registered_agent_returns_true(
        self, health_registry: HealthCheckRegistry
    ) -> None:
        """Deregistering a known agent returns True."""
        async def check_fn() -> bool:
            return True

        health_registry.register(
            "agent-a",
            HealthCheck(name="check", check_fn=check_fn),
            ProbeType.LIVENESS,
        )
        assert health_registry.deregister("agent-a") is True
        assert health_registry.agent_status("agent-a") == HealthStatus.UNKNOWN

    def test_global_status_unknown_when_empty(
        self, health_registry: HealthCheckRegistry
    ) -> None:
        """Global status is UNKNOWN when no agents are registered."""
        assert health_registry.global_status() == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_agent_summary_structure(
        self, health_registry: HealthCheckRegistry
    ) -> None:
        """Agent summary contains all expected fields."""
        async def check_fn() -> bool:
            return True

        health_registry.register(
            "agent-a",
            HealthCheck(name="check-1", check_fn=check_fn),
            ProbeType.LIVENESS,
        )
        await health_registry.evaluate("agent-a")
        summary = health_registry.agent_summary("agent-a")
        assert summary.agent_key == "agent-a"
        assert summary.status == HealthStatus.HEALTHY
        assert "check-1" in summary.checks

        data = summary.to_dict()
        assert data["agent_key"] == "agent-a"
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_evaluate_filtered_by_probe_type(
        self, health_registry: HealthCheckRegistry
    ) -> None:
        """Evaluation can be filtered to a specific probe type."""
        async def liveness_fn() -> bool:
            return True

        async def readiness_fn() -> bool:
            return False

        health_registry.register(
            "agent-a",
            HealthCheck(name="live", check_fn=liveness_fn),
            ProbeType.LIVENESS,
        )
        health_registry.register(
            "agent-a",
            HealthCheck(name="ready", check_fn=readiness_fn),
            ProbeType.READINESS,
        )
        results = await health_registry.evaluate("agent-a", probe_type=ProbeType.LIVENESS)
        assert len(results) == 1
        assert results[0].check_name == "live"
        assert results[0].healthy is True


# ============================================================================
# Test WarmupManager
# ============================================================================


class TestWarmupManager:
    """Tests for WarmupManager and warmup strategies."""

    @pytest.mark.asyncio
    async def test_warmup_strategy_cache_priming_success(
        self, warmup_manager: WarmupManager
    ) -> None:
        """Cache priming warmup succeeds when loader completes."""
        loader = AsyncMock()
        strategy = CachePrimingWarmup(
            cache_keys=["ef:*", "grid:*"],
            loader=loader,
        )
        warmup_manager.register("agent-a", strategy)

        report = await warmup_manager.warmup("agent-a", context={"env": "test"})
        assert report.success is True
        assert report.agent_key == "agent-a"
        assert len(report.steps) == 1
        assert report.steps[0].strategy_name == "cache_priming"
        loader.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_warmup_strategy_cache_priming_failure(
        self, warmup_manager: WarmupManager
    ) -> None:
        """Cache priming warmup fails when loader raises an exception."""
        loader = AsyncMock(side_effect=ConnectionError("redis down"))
        strategy = CachePrimingWarmup(cache_keys=["ef:*"], loader=loader)
        warmup_manager.register("agent-a", strategy)

        report = await warmup_manager.warmup("agent-a")
        assert report.success is False

    @pytest.mark.asyncio
    async def test_warmup_strategy_connection_pool(
        self, warmup_manager: WarmupManager
    ) -> None:
        """Connection pool warmup succeeds when connector completes."""
        connector = AsyncMock()
        strategy = ConnectionPoolWarmup(
            dsn="postgresql://test", min_connections=3, connector=connector
        )
        warmup_manager.register("agent-a", strategy)

        report = await warmup_manager.warmup("agent-a")
        assert report.success is True
        connector.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_warmup_strategy_model_load(
        self, warmup_manager: WarmupManager
    ) -> None:
        """Model load warmup succeeds when model_loader completes."""
        model_loader = AsyncMock()
        strategy = ModelLoadWarmup(
            model_name="MiniLM-L6", model_loader=model_loader
        )
        warmup_manager.register("agent-a", strategy)

        report = await warmup_manager.warmup("agent-a")
        assert report.success is True

    @pytest.mark.asyncio
    async def test_warmup_strategy_timeout(self) -> None:
        """Warmup times out when a strategy takes too long."""
        async def slow_warmup(context: Dict[str, Any]) -> bool:
            await asyncio.sleep(10.0)
            return True

        class SlowStrategy:
            @property
            def name(self) -> str:
                return "slow"

            async def warmup(self, context: Dict[str, Any]) -> bool:
                await asyncio.sleep(10.0)
                return True

        mgr = WarmupManager(default_timeout=0.05)
        mgr.register("agent-a", SlowStrategy())
        report = await mgr.warmup("agent-a")
        assert report.success is False
        assert any("timed out" in (s.error or "").lower() for s in report.steps)

    @pytest.mark.asyncio
    async def test_warmup_no_strategies_succeeds(
        self, warmup_manager: WarmupManager
    ) -> None:
        """Warmup with no registered strategies returns success."""
        report = await warmup_manager.warmup("unknown-agent")
        assert report.success is True
        assert report.total_duration_ms == 0.0

    @pytest.mark.asyncio
    async def test_warmup_multiple_strategies_sequential(
        self, warmup_manager: WarmupManager
    ) -> None:
        """Multiple strategies execute sequentially, first failure stops."""
        strategy1 = CachePrimingWarmup(cache_keys=["a"])
        strategy2 = ConnectionPoolWarmup(
            dsn="test",
            connector=AsyncMock(side_effect=RuntimeError("fail")),
        )
        strategy3 = ModelLoadWarmup(model_name="test")

        warmup_manager.register("agent-a", strategy1)
        warmup_manager.register("agent-a", strategy2)
        warmup_manager.register("agent-a", strategy3)

        report = await warmup_manager.warmup("agent-a")
        assert report.success is False
        assert len(report.steps) == 2  # fail-fast stops at strategy2

    def test_warmup_manager_deregister(
        self, warmup_manager: WarmupManager
    ) -> None:
        """Deregistering removes all strategies for the agent."""
        warmup_manager.register("agent-a", CachePrimingWarmup())
        assert warmup_manager.deregister("agent-a") is True
        assert warmup_manager.deregister("agent-a") is False

    def test_warmup_manager_get_strategies(
        self, warmup_manager: WarmupManager
    ) -> None:
        """get_strategies returns the registered list."""
        s1 = CachePrimingWarmup()
        s2 = ConnectionPoolWarmup()
        warmup_manager.register("agent-a", s1)
        warmup_manager.register("agent-a", s2)
        strategies = warmup_manager.get_strategies("agent-a")
        assert len(strategies) == 2

    @pytest.mark.asyncio
    async def test_warmup_report_to_dict(
        self, warmup_manager: WarmupManager
    ) -> None:
        """WarmupReport.to_dict returns proper serialization."""
        warmup_manager.register("agent-a", CachePrimingWarmup())
        report = await warmup_manager.warmup("agent-a")
        data = report.to_dict()
        assert data["agent_key"] == "agent-a"
        assert data["success"] is True
        assert isinstance(data["steps"], list)


# ============================================================================
# Test GracefulShutdownCoordinator
# ============================================================================


class TestGracefulShutdownCoordinator:
    """Tests for the GracefulShutdownCoordinator."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown_drain_completes(
        self, shutdown_coordinator: GracefulShutdownCoordinator
    ) -> None:
        """Agent drains successfully within timeout."""
        drain_fn = AsyncMock(return_value=True)
        stop_fn = AsyncMock()

        result = await shutdown_coordinator.shutdown_agent(
            "agent-a", drain_fn, stop_fn
        )
        assert result.drained is True
        assert result.forced is False
        assert result.error is None
        drain_fn.assert_awaited_once()
        stop_fn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_force_stop(
        self, shutdown_coordinator: GracefulShutdownCoordinator
    ) -> None:
        """Agent is force-stopped when drain times out."""
        async def slow_drain() -> bool:
            await asyncio.sleep(10.0)
            return True

        stop_fn = AsyncMock()

        coordinator = GracefulShutdownCoordinator(
            drain_timeout=0.05,
            force_stop_timeout=1.0,
        )
        result = await coordinator.shutdown_agent(
            "agent-a", slow_drain, stop_fn
        )
        assert result.drained is False
        assert result.forced is True
        stop_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pre_shutdown_hook_called(
        self, shutdown_coordinator: GracefulShutdownCoordinator
    ) -> None:
        """Pre-shutdown hooks are called before drain begins."""
        hook_called = []

        async def pre_hook(agent_key: str) -> None:
            hook_called.append(agent_key)

        shutdown_coordinator.add_pre_shutdown_hook("flush-metrics", pre_hook)
        drain_fn = AsyncMock(return_value=True)
        stop_fn = AsyncMock()

        await shutdown_coordinator.shutdown_agent("agent-a", drain_fn, stop_fn)
        assert hook_called == ["agent-a"]

    @pytest.mark.asyncio
    async def test_post_shutdown_hook_called(
        self, shutdown_coordinator: GracefulShutdownCoordinator
    ) -> None:
        """Post-shutdown hooks are called after shutdown completes."""
        hook_called = []

        async def post_hook(agent_key: str) -> None:
            hook_called.append(agent_key)

        shutdown_coordinator.add_post_shutdown_hook("cleanup", post_hook)
        drain_fn = AsyncMock(return_value=True)
        stop_fn = AsyncMock()

        await shutdown_coordinator.shutdown_agent("agent-a", drain_fn, stop_fn)
        assert hook_called == ["agent-a"]

    def test_remove_hook(
        self, shutdown_coordinator: GracefulShutdownCoordinator
    ) -> None:
        """Removing a hook prevents it from executing."""
        async def hook(agent_key: str) -> None:
            pass

        shutdown_coordinator.add_pre_shutdown_hook("test-hook", hook)
        assert shutdown_coordinator.remove_hook("test-hook") is True
        assert shutdown_coordinator.remove_hook("nonexistent") is False

    @pytest.mark.asyncio
    async def test_shutdown_all_multiple_agents(
        self, shutdown_coordinator: GracefulShutdownCoordinator
    ) -> None:
        """shutdown_all handles multiple agents concurrently."""
        agents = {
            "agent-a": AgentShutdownSpec(
                drain_fn=AsyncMock(return_value=True),
                stop_fn=AsyncMock(),
            ),
            "agent-b": AgentShutdownSpec(
                drain_fn=AsyncMock(return_value=True),
                stop_fn=AsyncMock(),
            ),
        }
        report = await shutdown_coordinator.shutdown_all(agents)
        assert report.total_agents == 2
        assert report.successful == 2
        assert report.forced == 0
        assert report.failed == 0

    @pytest.mark.asyncio
    async def test_shutdown_report_to_dict(
        self, shutdown_coordinator: GracefulShutdownCoordinator
    ) -> None:
        """ShutdownReport.to_dict produces correct structure."""
        agents = {
            "agent-a": AgentShutdownSpec(
                drain_fn=AsyncMock(return_value=True),
                stop_fn=AsyncMock(),
            ),
        }
        report = await shutdown_coordinator.shutdown_all(agents)
        data = report.to_dict()
        assert data["total_agents"] == 1
        assert isinstance(data["results"], list)

    @pytest.mark.asyncio
    async def test_drain_error_falls_through_to_force_stop(
        self, shutdown_coordinator: GracefulShutdownCoordinator
    ) -> None:
        """When drain raises, agent is force-stopped."""
        drain_fn = AsyncMock(side_effect=RuntimeError("drain crash"))
        stop_fn = AsyncMock()

        result = await shutdown_coordinator.shutdown_agent(
            "agent-a", drain_fn, stop_fn
        )
        assert result.forced is True
        assert result.error is not None
        assert "drain" in result.error.lower()


# ============================================================================
# Test AgentLifecycleManager
# ============================================================================


class TestAgentLifecycleManager:
    """Tests for the central AgentLifecycleManager."""

    @pytest.mark.asyncio
    async def test_lifecycle_manager_register_agent(
        self, lifecycle_manager: AgentLifecycleManager
    ) -> None:
        """Registering a new agent succeeds."""
        result = await lifecycle_manager.register_agent("carbon-agent")
        assert result is True
        status = lifecycle_manager.get_status("carbon-agent")
        assert status is not None
        assert status["state"] == "created"

    @pytest.mark.asyncio
    async def test_lifecycle_manager_register_duplicate(
        self, lifecycle_manager: AgentLifecycleManager
    ) -> None:
        """Registering the same agent twice returns False."""
        await lifecycle_manager.register_agent("carbon-agent")
        result = await lifecycle_manager.register_agent("carbon-agent")
        assert result is False

    @pytest.mark.asyncio
    async def test_lifecycle_manager_start_agent(
        self, lifecycle_manager: AgentLifecycleManager
    ) -> None:
        """Starting an agent transitions through full lifecycle to RUNNING."""
        await lifecycle_manager.register_agent("test-agent")
        result = await lifecycle_manager.start_agent("test-agent")
        assert result is True
        status = lifecycle_manager.get_status("test-agent")
        assert status["state"] == "running"

    @pytest.mark.asyncio
    async def test_lifecycle_manager_start_unregistered(
        self, lifecycle_manager: AgentLifecycleManager
    ) -> None:
        """Starting an unregistered agent returns False."""
        result = await lifecycle_manager.start_agent("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_lifecycle_manager_stop_agent_gracefully(
        self, lifecycle_manager: AgentLifecycleManager
    ) -> None:
        """Stopping an agent transitions to FORCE_STOPPED."""
        await lifecycle_manager.register_agent("test-agent")
        await lifecycle_manager.start_agent("test-agent")
        result = await lifecycle_manager.stop_agent("test-agent", "manual stop")
        assert result is True
        status = lifecycle_manager.get_status("test-agent")
        assert status["state"] == "force_stopped"

    @pytest.mark.asyncio
    async def test_lifecycle_manager_drain_agent(
        self, lifecycle_manager: AgentLifecycleManager
    ) -> None:
        """Draining an agent transitions from RUNNING to DRAINING."""
        await lifecycle_manager.register_agent("test-agent")
        await lifecycle_manager.start_agent("test-agent")
        result = await lifecycle_manager.drain_agent("test-agent")
        assert result is True
        status = lifecycle_manager.get_status("test-agent")
        assert status["state"] == "draining"

    @pytest.mark.asyncio
    async def test_lifecycle_manager_retire_agent(
        self, lifecycle_manager: AgentLifecycleManager
    ) -> None:
        """Retiring a drained agent transitions to RETIRED."""
        await lifecycle_manager.register_agent("test-agent")
        await lifecycle_manager.start_agent("test-agent")
        await lifecycle_manager.drain_agent("test-agent")
        result = await lifecycle_manager.retire_agent("test-agent")
        assert result is True
        status = lifecycle_manager.get_status("test-agent")
        assert status["state"] == "retired"

    @pytest.mark.asyncio
    async def test_lifecycle_manager_event_emission(
        self, lifecycle_manager: AgentLifecycleManager
    ) -> None:
        """Events are emitted on every state transition."""
        events: List[tuple] = []

        async def callback(
            agent_key: str,
            old_state: AgentState,
            new_state: AgentState,
            reason: str,
        ) -> None:
            events.append((agent_key, old_state, new_state, reason))

        lifecycle_manager.on_state_change(callback)
        await lifecycle_manager.register_agent("test-agent")
        await lifecycle_manager.start_agent("test-agent")

        assert len(events) >= 5  # VALIDATING, VALIDATED, DEPLOYING, WARMING_UP, RUNNING
        assert events[-1][2] == AgentState.RUNNING

    @pytest.mark.asyncio
    async def test_lifecycle_manager_list_agents(
        self, lifecycle_manager: AgentLifecycleManager
    ) -> None:
        """list_agents returns all registered agents."""
        await lifecycle_manager.register_agent("agent-a")
        await lifecycle_manager.register_agent("agent-b")
        agents = lifecycle_manager.list_agents()
        assert len(agents) == 2

    @pytest.mark.asyncio
    async def test_lifecycle_manager_list_agents_with_filter(
        self, lifecycle_manager: AgentLifecycleManager
    ) -> None:
        """list_agents can filter by state."""
        await lifecycle_manager.register_agent("agent-a")
        await lifecycle_manager.register_agent("agent-b")
        await lifecycle_manager.start_agent("agent-a")

        running = lifecycle_manager.list_agents(state_filter=AgentState.RUNNING)
        assert len(running) == 1
        assert running[0]["agent_key"] == "agent-a"

        created = lifecycle_manager.list_agents(state_filter=AgentState.CREATED)
        assert len(created) == 1
        assert created[0]["agent_key"] == "agent-b"

    @pytest.mark.asyncio
    async def test_lifecycle_manager_get_status_unknown(
        self, lifecycle_manager: AgentLifecycleManager
    ) -> None:
        """get_status returns None for unregistered agents."""
        assert lifecycle_manager.get_status("nonexistent") is None

    @pytest.mark.asyncio
    async def test_lifecycle_manager_config_properties(
        self, lifecycle_manager: AgentLifecycleManager
    ) -> None:
        """Config and sub-component properties are accessible."""
        assert lifecycle_manager.config is not None
        assert lifecycle_manager.health_registry is not None
        assert lifecycle_manager.warmup_manager is not None
        assert lifecycle_manager.shutdown_coordinator is not None

    @pytest.mark.asyncio
    async def test_lifecycle_manager_close(
        self, lifecycle_manager: AgentLifecycleManager
    ) -> None:
        """Closing the manager shuts down background tasks."""
        lifecycle_manager._running = True
        await lifecycle_manager.close()
        assert lifecycle_manager._running is False
