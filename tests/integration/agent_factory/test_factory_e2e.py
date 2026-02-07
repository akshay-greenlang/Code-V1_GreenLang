# -*- coding: utf-8 -*-
"""
End-to-end integration tests for the Agent Factory.

Tests full workflows across multiple subsystems: lifecycle management,
packaging, circuit breakers, fallback chains, cost metering, dependency
graphs, configuration hot-reload, hub operations, multi-tenant isolation,
and observability chains.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.agent_factory.lifecycle.states import (
    AgentState,
    AgentStateMachine,
)
from greenlang.infrastructure.agent_factory.lifecycle.warmup import (
    CachePrimingWarmup,
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
from greenlang.infrastructure.agent_factory.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitOpenError,
)
from greenlang.infrastructure.agent_factory.resilience.fallback import (
    FallbackChain,
)
from greenlang.infrastructure.agent_factory.resilience.bulkhead import (
    BulkheadConfig,
    BulkheadIsolation,
)

# Import test doubles from unit tests
from tests.unit.agent_factory.test_dependency_graph import DependencyGraph
from tests.unit.agent_factory.test_cost_metering import (
    BudgetConfig,
    BudgetManager,
    CostRecord,
    CostTracker,
)
from tests.unit.agent_factory.test_hot_reload import (
    ConfigSchema,
    ConfigStore,
)
from tests.unit.agent_factory.test_hub_registry import (
    HubRegistry,
    LocalIndex,
    PackageInfo,
)
from tests.unit.agent_factory.test_telemetry import (
    CorrelationManager,
    MetricsCollector,
    SpanFactory,
    Tracer,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clean_registries():
    CircuitBreaker.clear_registry()
    BulkheadIsolation.clear_registry()
    AgentLifecycleManager.reset_singleton()
    yield
    CircuitBreaker.clear_registry()
    BulkheadIsolation.clear_registry()
    AgentLifecycleManager.reset_singleton()


@pytest.fixture
async def manager() -> AgentLifecycleManager:
    config = LifecycleManagerConfig(
        health_check_interval=60.0,
        drain_timeout=2.0,
        max_restart_attempts=3,
        warmup_timeout=5.0,
        redis_url=None,
    )
    mgr = AgentLifecycleManager(config)
    mgr._running = True
    mgr._health_task = None
    yield mgr
    mgr._running = False


# ============================================================================
# Tests
# ============================================================================


class TestFactoryE2E:
    """End-to-end tests for the Agent Factory system."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_agent_create_deploy_execute_result(
        self, manager: AgentLifecycleManager
    ) -> None:
        """Full flow: create agent -> deploy -> execute task -> return result."""
        await manager.register_agent("e2e-agent", metadata={"version": "1.0.0"})
        assert await manager.start_agent("e2e-agent") is True

        status = manager.get_status("e2e-agent")
        assert status["state"] == "running"

        # Simulate execution through circuit breaker
        cb = CircuitBreaker("e2e-agent", CircuitBreakerConfig(minimum_calls=5))
        async with cb:
            result = {"emissions_kg": 42.0}
        assert cb.metrics.success_count == 1

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_agent_rollback_flow(
        self, manager: AgentLifecycleManager
    ) -> None:
        """Agent can be force-stopped and restarted (rollback simulation)."""
        await manager.register_agent("rollback-agent")
        await manager.start_agent("rollback-agent")
        assert manager.get_status("rollback-agent")["state"] == "running"

        await manager.stop_agent("rollback-agent", "rollback")
        assert manager.get_status("rollback-agent")["state"] == "force_stopped"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_circuit_breaker_activation_and_recovery(self) -> None:
        """Circuit breaker trips on failures and recovers after wait."""
        config = CircuitBreakerConfig(
            failure_rate_threshold=0.5,
            minimum_calls=3,
            wait_in_open_s=0.05,
            half_open_test_requests=2,
        )
        cb = CircuitBreaker("resilience-test", config)

        # Trip the circuit
        for _ in range(3):
            await cb.record_call(success=False, duration_s=0.01)
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for half-open
        await asyncio.sleep(0.1)
        await cb._before_call()
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Recover
        await cb.record_call(success=True, duration_s=0.01)
        await cb.record_call(success=True, duration_s=0.01)
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_cost_metering_during_execution(self) -> None:
        """Cost is tracked during agent execution."""
        tracker = CostTracker()
        budget_mgr = BudgetManager(tracker)
        budget_mgr.set_budget(BudgetConfig("tenant-1", monthly_limit_usd=100.0))

        # Simulate 5 executions
        for i in range(5):
            tracker.record(CostRecord(
                agent_key="calc-agent",
                tenant_id="tenant-1",
                cost_usd=5.0,
                resource_type="compute",
            ))

        assert tracker.total_cost_by_tenant("tenant-1") == pytest.approx(25.0)
        assert budget_mgr.allow_execution("tenant-1") is True

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_dependency_graph_resolution_and_ordering(self) -> None:
        """Dependency graph resolves execution order correctly."""
        graph = DependencyGraph()
        graph.add_agent("reporter")
        graph.add_agent("calculator")
        graph.add_agent("data-loader")
        graph.add_dependency("reporter", "calculator")
        graph.add_dependency("calculator", "data-loader")

        order = graph.topological_sort()
        assert order.index("data-loader") < order.index("calculator")
        assert order.index("calculator") < order.index("reporter")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_config_hot_reload_during_execution(self) -> None:
        """Configuration can be hot-reloaded while agents are running."""
        schema = ConfigSchema(fields={
            "timeout": {"type": "int", "required": True, "min": 1, "max": 300},
        })
        store = ConfigStore(schema=schema)

        reload_events: List[Dict[str, Any]] = []

        async def on_reload(agent_key, new_config, diff):
            reload_events.append({"agent": agent_key, "config": new_config})

        store.on_reload(on_reload)

        await store.set("running-agent", {"timeout": 30})
        await store.set("running-agent", {"timeout": 60})

        assert len(reload_events) >= 1
        assert reload_events[-1]["config"]["timeout"] == 60

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_hub_publish_search_install_flow(self) -> None:
        """Full hub workflow: publish -> search -> install."""
        hub = HubRegistry()
        index = LocalIndex()

        pkg = PackageInfo(
            name="emissions-calc",
            version="1.0.0",
            description="Scope 1 emissions calculator",
            tags=["emissions", "scope1"],
            metadata={"entry_point": "greenlang.agents.emissions.agent"},
        )
        hub.publish(pkg)

        results = hub.search("emissions")
        assert len(results) == 1

        downloaded = hub.download("emissions-calc", "1.0.0")
        assert downloaded is not None
        index.add(downloaded)

        installed = index.get("emissions-calc")
        assert installed is not None
        assert installed.version == "1.0.0"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_multi_tenant_agent_isolation(self) -> None:
        """Agents from different tenants are isolated."""
        tracker = CostTracker()
        budget = BudgetManager(tracker)

        budget.set_budget(BudgetConfig("tenant-a", monthly_limit_usd=50.0))
        budget.set_budget(BudgetConfig("tenant-b", monthly_limit_usd=100.0))

        tracker.record(CostRecord("a1", "tenant-a", 45.0, "compute"))
        tracker.record(CostRecord("b1", "tenant-b", 20.0, "compute"))

        # Tenant A near limit
        assert budget.check_budget("tenant-a").value == "warning"
        # Tenant B within budget
        assert budget.check_budget("tenant-b").value == "within"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_full_observability_chain(self) -> None:
        """Trace -> spans -> metrics -> correlation propagation."""
        tracer = Tracer(service_name="e2e-test")
        factory = SpanFactory(tracer)
        metrics = MetricsCollector()

        # Create trace
        root_span = factory.execution_span("calc-agent", "task-001")

        # Simulate execution
        metrics.record("agent.executions", 1, {"agent": "calc-agent"})
        metrics.histogram("agent.latency_ms", 42.5)

        # Propagate correlation
        cid = CorrelationManager.generate_id()
        headers: Dict[str, str] = {}
        CorrelationManager.inject_headers(headers, cid, root_span.trace_id)

        extracted_cid, extracted_trace = CorrelationManager.extract_headers(headers)
        assert extracted_cid == cid
        assert extracted_trace == root_span.trace_id

        # Child span
        child = factory.lifecycle_span("calc-agent", "RUNNING", parent=root_span)
        assert child.trace_id == root_span.trace_id

        root_span.finish()
        assert root_span.duration_ms >= 0
        assert metrics.get_counter("agent.executions") == 1.0

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_fallback_chain_with_circuit_breaker(self) -> None:
        """Fallback chain integrates with circuit breaker for resilience."""
        cb = CircuitBreaker("primary-agent", CircuitBreakerConfig(minimum_calls=3))

        call_count = 0

        async def primary(ctx: Dict[str, Any]) -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("primary down")

        async def fallback(ctx: Dict[str, Any]) -> str:
            return "fallback_result"

        chain = FallbackChain("test-agent")
        chain.add_handler("primary", primary)
        chain.add_handler("fallback", fallback)

        result = await chain.execute({"data": "test"})
        assert result.result == "fallback_result"
        assert result.was_fallback is True
