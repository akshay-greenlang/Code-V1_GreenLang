# -*- coding: utf-8 -*-
"""
Unit tests for Engine 4: Parallel Execution Engine -- AGENT-EUDR-026

Tests DAG-aware parallel agent execution including dependency resolution,
slot acquisition/release, execution queue management with priority ordering,
running agent status tracking, circuit breaker integration, ETA estimation,
and workflow cleanup operations.

Test count: ~95 tests
Author: GreenLang Platform Team
Date: March 2026
"""

import time
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.due_diligence_orchestrator.parallel_execution_engine import (
    ParallelExecutionEngine,
    ExecutionSlot,
    ReadyAgent,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    AgentNode,
    WorkflowEdge,
    WorkflowDefinition,
    WorkflowType,
    EUDRCommodity,
    DueDiligencePhase,
    CircuitBreakerState,
    AgentExecutionStatus,
    PHASE_1_AGENTS,
    PHASE_2_AGENTS,
    ALL_EUDR_AGENTS,
)
from greenlang.agents.eudr.due_diligence_orchestrator.config import (
    DueDiligenceOrchestratorConfig,
    get_config,
    set_config,
    reset_config,
)

from tests.agents.eudr.due_diligence_orchestrator.conftest import (
    STANDARD_WORKFLOW_EDGES,
    _make_agent_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_definition(
    nodes_spec: list[tuple[str, int]],
    edges_spec: list[tuple[str, str]],
) -> WorkflowDefinition:
    """Build a minimal WorkflowDefinition from specs.

    Args:
        nodes_spec: List of (agent_id, layer) tuples.
        edges_spec: List of (source, target) tuples.
    """
    nodes = [
        _make_agent_node(aid, layer=layer)
        for aid, layer in nodes_spec
    ]
    edges = [
        WorkflowEdge(source=s, target=t)
        for s, t in edges_spec
    ]
    return WorkflowDefinition(
        definition_id="def-test",
        name="Test Workflow",
        nodes=nodes,
        edges=edges,
    )


# =========================================================================
# ExecutionSlot tests
# =========================================================================


class TestExecutionSlot:
    """Test ExecutionSlot dataclass behavior."""

    def test_init_stores_agent_id(self):
        slot = ExecutionSlot(
            agent_id="EUDR-001",
            workflow_id="wf-001",
            timeout_s=120,
        )
        assert slot.agent_id == "EUDR-001"

    def test_init_stores_workflow_id(self):
        slot = ExecutionSlot(
            agent_id="EUDR-001",
            workflow_id="wf-001",
            timeout_s=120,
        )
        assert slot.workflow_id == "wf-001"

    def test_init_stores_timeout(self):
        slot = ExecutionSlot(
            agent_id="EUDR-001",
            workflow_id="wf-001",
            timeout_s=60,
        )
        assert slot.timeout_s == 60

    def test_init_default_layer_is_zero(self):
        slot = ExecutionSlot(
            agent_id="EUDR-001",
            workflow_id="wf-001",
            timeout_s=120,
        )
        assert slot.layer == 0

    def test_init_custom_layer(self):
        slot = ExecutionSlot(
            agent_id="EUDR-002",
            workflow_id="wf-001",
            timeout_s=120,
            layer=3,
        )
        assert slot.layer == 3

    def test_started_at_is_set_on_init(self):
        # _utcnow() zeroes microseconds, so align before/after accordingly
        before = datetime.now(timezone.utc).replace(microsecond=0)
        slot = ExecutionSlot(
            agent_id="EUDR-001",
            workflow_id="wf-001",
            timeout_s=120,
        )
        after = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=1)
        assert before <= slot.started_at <= after

    def test_is_timed_out_returns_false_when_fresh(self):
        slot = ExecutionSlot(
            agent_id="EUDR-001",
            workflow_id="wf-001",
            timeout_s=300,
        )
        assert slot.is_timed_out() is False

    def test_is_timed_out_returns_true_when_expired(self):
        slot = ExecutionSlot(
            agent_id="EUDR-001",
            workflow_id="wf-001",
            timeout_s=120,
        )
        # Manually backdate started_at to simulate timeout
        slot.started_at = datetime.now(timezone.utc) - timedelta(seconds=200)
        assert slot.is_timed_out() is True

    def test_is_timed_out_boundary_not_timed_out(self):
        """Just under timeout should not be timed out."""
        slot = ExecutionSlot(
            agent_id="EUDR-001",
            workflow_id="wf-001",
            timeout_s=10,
        )
        slot.started_at = datetime.now(timezone.utc) - timedelta(seconds=9)
        assert slot.is_timed_out() is False


# =========================================================================
# ReadyAgent tests
# =========================================================================


class TestReadyAgent:
    """Test ReadyAgent priority queue entry."""

    def test_init_stores_agent_id(self):
        ra = ReadyAgent(agent_id="EUDR-001", workflow_id="wf-001")
        assert ra.agent_id == "EUDR-001"

    def test_init_stores_workflow_id(self):
        ra = ReadyAgent(agent_id="EUDR-001", workflow_id="wf-001")
        assert ra.workflow_id == "wf-001"

    def test_init_default_layer(self):
        ra = ReadyAgent(agent_id="EUDR-001", workflow_id="wf-001")
        assert ra.layer == 0

    def test_init_default_priority(self):
        ra = ReadyAgent(agent_id="EUDR-001", workflow_id="wf-001")
        assert ra.priority == 3

    def test_init_default_timeout(self):
        ra = ReadyAgent(agent_id="EUDR-001", workflow_id="wf-001")
        assert ra.timeout_s == 120

    def test_init_custom_values(self):
        ra = ReadyAgent(
            agent_id="EUDR-016",
            workflow_id="wf-002",
            layer=2,
            priority=1,
            timeout_s=60,
        )
        assert ra.layer == 2
        assert ra.priority == 1
        assert ra.timeout_s == 60

    def test_enqueued_at_is_set(self):
        # _utcnow() zeroes microseconds, so align before/after accordingly
        before = datetime.now(timezone.utc).replace(microsecond=0)
        ra = ReadyAgent(agent_id="EUDR-001", workflow_id="wf-001")
        after = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=1)
        assert before <= ra.enqueued_at <= after

    def test_sort_key_returns_tuple(self):
        ra = ReadyAgent(
            agent_id="EUDR-001",
            workflow_id="wf-001",
            layer=1,
            priority=2,
        )
        key = ra.sort_key()
        assert isinstance(key, tuple)
        assert len(key) == 3
        assert key[0] == 1  # layer
        assert key[1] == 2  # priority

    def test_sort_key_layer_first(self):
        """Lower layer sorts first."""
        ra_layer0 = ReadyAgent(
            agent_id="A", workflow_id="wf-001", layer=0, priority=5,
        )
        ra_layer1 = ReadyAgent(
            agent_id="B", workflow_id="wf-001", layer=1, priority=1,
        )
        assert ra_layer0.sort_key() < ra_layer1.sort_key()

    def test_sort_key_priority_second(self):
        """Within same layer, lower priority number (higher priority) sorts first."""
        ra_high = ReadyAgent(
            agent_id="A", workflow_id="wf-001", layer=0, priority=1,
        )
        ra_low = ReadyAgent(
            agent_id="B", workflow_id="wf-001", layer=0, priority=5,
        )
        assert ra_high.sort_key() < ra_low.sort_key()

    def test_sort_key_enqueued_at_third(self):
        """Within same layer and priority, earlier enqueue wins."""
        # _utcnow() zeroes microseconds, so mock to control time difference
        t1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc)
        with patch(
            "greenlang.agents.eudr.due_diligence_orchestrator.parallel_execution_engine._utcnow",
            return_value=t1,
        ):
            ra_first = ReadyAgent(
                agent_id="A", workflow_id="wf-001", layer=0, priority=3,
            )
        with patch(
            "greenlang.agents.eudr.due_diligence_orchestrator.parallel_execution_engine._utcnow",
            return_value=t2,
        ):
            ra_second = ReadyAgent(
                agent_id="B", workflow_id="wf-001", layer=0, priority=3,
            )
        assert ra_first.sort_key() < ra_second.sort_key()


# =========================================================================
# ParallelExecutionEngine initialization
# =========================================================================


class TestParallelExecutionEngineInit:
    """Test engine initialization."""

    def test_init_with_default_config(self, default_config):
        engine = ParallelExecutionEngine()
        assert engine is not None

    def test_init_with_explicit_config(self, default_config):
        engine = ParallelExecutionEngine(config=default_config)
        assert engine._config is default_config

    def test_init_empty_state(self, default_config):
        engine = ParallelExecutionEngine(config=default_config)
        assert engine.get_running_count() == 0
        assert engine.get_queue_size() == 0

    def test_init_with_custom_concurrency(self, default_config):
        cfg = DueDiligenceOrchestratorConfig(
            max_concurrent_agents=5,
            global_concurrency_limit=20,
        )
        set_config(cfg)
        engine = ParallelExecutionEngine(config=cfg)
        assert engine._config.max_concurrent_agents == 5
        assert engine._config.global_concurrency_limit == 20


# =========================================================================
# get_ready_agents: DAG dependency resolution
# =========================================================================


class TestGetReadyAgents:
    """Test DAG-aware agent readiness evaluation."""

    def test_root_agent_is_ready_when_no_deps(self, parallel_execution_engine):
        """Root node with no dependencies should be ready."""
        defn = _simple_definition(
            nodes_spec=[("A", 0), ("B", 1)],
            edges_spec=[("A", "B")],
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed=set(),
            running=set(),
        )
        assert "A" in ready

    def test_dependent_agent_not_ready_until_deps_complete(
        self, parallel_execution_engine,
    ):
        defn = _simple_definition(
            nodes_spec=[("A", 0), ("B", 1)],
            edges_spec=[("A", "B")],
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed=set(),
            running=set(),
        )
        assert "B" not in ready

    def test_dependent_agent_ready_after_dep_completes(
        self, parallel_execution_engine,
    ):
        defn = _simple_definition(
            nodes_spec=[("A", 0), ("B", 1)],
            edges_spec=[("A", "B")],
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed={"A"},
            running=set(),
        )
        assert "B" in ready

    def test_multi_dep_requires_all_deps(self, parallel_execution_engine):
        """Agent with two dependencies needs both completed."""
        defn = _simple_definition(
            nodes_spec=[("A", 0), ("B", 0), ("C", 1)],
            edges_spec=[("A", "C"), ("B", "C")],
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed={"A"},
            running=set(),
        )
        assert "C" not in ready

    def test_multi_dep_all_satisfied(self, parallel_execution_engine):
        defn = _simple_definition(
            nodes_spec=[("A", 0), ("B", 0), ("C", 1)],
            edges_spec=[("A", "C"), ("B", "C")],
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed={"A", "B"},
            running=set(),
        )
        assert "C" in ready

    def test_running_agents_excluded(self, parallel_execution_engine):
        defn = _simple_definition(
            nodes_spec=[("A", 0), ("B", 0)],
            edges_spec=[],
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed=set(),
            running={"A"},
        )
        assert "A" not in ready
        assert "B" in ready

    def test_completed_agents_excluded(self, parallel_execution_engine):
        defn = _simple_definition(
            nodes_spec=[("A", 0), ("B", 0)],
            edges_spec=[],
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed={"A"},
            running=set(),
        )
        assert "A" not in ready

    def test_skipped_agents_satisfy_deps(self, parallel_execution_engine):
        """Skipped agents count as 'done' for dependency resolution."""
        defn = _simple_definition(
            nodes_spec=[("A", 0), ("B", 1)],
            edges_spec=[("A", "B")],
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed=set(),
            running=set(),
            skipped={"A"},
        )
        assert "B" in ready

    def test_failed_agents_satisfy_deps(self, parallel_execution_engine):
        """Failed agents count as 'done' for dependency resolution."""
        defn = _simple_definition(
            nodes_spec=[("A", 0), ("B", 1)],
            edges_spec=[("A", "B")],
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed=set(),
            running=set(),
            failed={"A"},
        )
        assert "B" in ready

    def test_concurrency_limit_caps_ready_agents(self, default_config):
        """Per-workflow concurrency caps the returned ready list."""
        cfg = DueDiligenceOrchestratorConfig(max_concurrent_agents=2)
        set_config(cfg)
        engine = ParallelExecutionEngine(config=cfg)

        defn = _simple_definition(
            nodes_spec=[("A", 0), ("B", 0), ("C", 0), ("D", 0)],
            edges_spec=[],
        )
        ready = engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed=set(),
            running=set(),
        )
        assert len(ready) == 2

    def test_concurrency_limit_accounts_for_running(self, default_config):
        """Available slots = max_concurrent - already_running."""
        cfg = DueDiligenceOrchestratorConfig(max_concurrent_agents=3)
        set_config(cfg)
        engine = ParallelExecutionEngine(config=cfg)

        defn = _simple_definition(
            nodes_spec=[("A", 0), ("B", 0), ("C", 0), ("D", 0)],
            edges_spec=[],
        )
        ready = engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed=set(),
            running={"A"},
        )
        # 3 - 1 running = 2 slots
        assert len(ready) == 2

    def test_circuit_breaker_open_excludes_agent(
        self, parallel_execution_engine,
    ):
        defn = _simple_definition(
            nodes_spec=[("A", 0), ("B", 0)],
            edges_spec=[],
        )
        parallel_execution_engine.set_circuit_breaker_state(
            "A", CircuitBreakerState.OPEN,
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed=set(),
            running=set(),
        )
        assert "A" not in ready
        assert "B" in ready

    def test_circuit_breaker_closed_allows_agent(
        self, parallel_execution_engine,
    ):
        defn = _simple_definition(
            nodes_spec=[("A", 0)],
            edges_spec=[],
        )
        parallel_execution_engine.set_circuit_breaker_state(
            "A", CircuitBreakerState.CLOSED,
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed=set(),
            running=set(),
        )
        assert "A" in ready

    def test_circuit_breaker_half_open_allows_agent(
        self, parallel_execution_engine,
    ):
        """HALF_OPEN is not OPEN, so agent should be allowed."""
        defn = _simple_definition(
            nodes_spec=[("A", 0)],
            edges_spec=[],
        )
        parallel_execution_engine.set_circuit_breaker_state(
            "A", CircuitBreakerState.HALF_OPEN,
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed=set(),
            running=set(),
        )
        assert "A" in ready

    def test_all_agents_completed_returns_empty(
        self, parallel_execution_engine,
    ):
        defn = _simple_definition(
            nodes_spec=[("A", 0), ("B", 1)],
            edges_spec=[("A", "B")],
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed={"A", "B"},
            running=set(),
        )
        assert ready == []

    def test_empty_workflow_returns_empty(self, parallel_execution_engine):
        defn = WorkflowDefinition(
            definition_id="def-empty",
            name="Empty",
            nodes=[],
            edges=[],
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed=set(),
            running=set(),
        )
        assert ready == []

    def test_ready_agents_sorted_by_layer_then_id(
        self, parallel_execution_engine,
    ):
        """Results should be sorted by (layer, agent_id)."""
        defn = _simple_definition(
            nodes_spec=[("C", 0), ("A", 0), ("B", 0)],
            edges_spec=[],
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed=set(),
            running=set(),
        )
        assert ready == sorted(ready)

    def test_standard_workflow_root_is_eudr001(
        self, parallel_execution_engine, standard_workflow_definition,
    ):
        """In the standard EUDR workflow, EUDR-001 is the root."""
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=standard_workflow_definition,
            completed=set(),
            running=set(),
        )
        assert "EUDR-001" in ready


# =========================================================================
# Slot management: acquire and release
# =========================================================================


class TestSlotManagement:
    """Test execution slot acquisition and release."""

    def test_acquire_slot_success(self, parallel_execution_engine):
        result = parallel_execution_engine.acquire_slot(
            agent_id="EUDR-001",
            workflow_id="wf-001",
        )
        assert result is True

    def test_acquire_slot_increments_running_count(
        self, parallel_execution_engine,
    ):
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        assert parallel_execution_engine.get_running_count() == 1

    def test_acquire_slot_with_custom_timeout(
        self, parallel_execution_engine,
    ):
        result = parallel_execution_engine.acquire_slot(
            agent_id="EUDR-001",
            workflow_id="wf-001",
            timeout_s=60,
        )
        assert result is True

    def test_acquire_slot_with_layer(self, parallel_execution_engine):
        result = parallel_execution_engine.acquire_slot(
            agent_id="EUDR-001",
            workflow_id="wf-001",
            layer=2,
        )
        assert result is True

    def test_acquire_duplicate_agent_fails(self, parallel_execution_engine):
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        result = parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        assert result is False

    def test_acquire_same_agent_different_workflow_succeeds(
        self, parallel_execution_engine,
    ):
        result1 = parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        result2 = parallel_execution_engine.acquire_slot("EUDR-001", "wf-002")
        assert result1 is True
        assert result2 is True
        assert parallel_execution_engine.get_running_count() == 2

    def test_acquire_per_workflow_limit(self, default_config):
        cfg = DueDiligenceOrchestratorConfig(max_concurrent_agents=2)
        set_config(cfg)
        engine = ParallelExecutionEngine(config=cfg)

        engine.acquire_slot("EUDR-001", "wf-001")
        engine.acquire_slot("EUDR-002", "wf-001")
        result = engine.acquire_slot("EUDR-003", "wf-001")
        assert result is False

    def test_acquire_per_workflow_limit_does_not_affect_other_workflows(
        self, default_config,
    ):
        cfg = DueDiligenceOrchestratorConfig(max_concurrent_agents=2)
        set_config(cfg)
        engine = ParallelExecutionEngine(config=cfg)

        engine.acquire_slot("EUDR-001", "wf-001")
        engine.acquire_slot("EUDR-002", "wf-001")
        # wf-002 is a different workflow, should succeed
        result = engine.acquire_slot("EUDR-001", "wf-002")
        assert result is True

    def test_acquire_global_limit(self, default_config):
        cfg = DueDiligenceOrchestratorConfig(
            max_concurrent_agents=25,
            global_concurrency_limit=25,
        )
        set_config(cfg)
        engine = ParallelExecutionEngine(config=cfg)

        # Acquire slots up to the global limit across different workflows
        for i in range(25):
            engine.acquire_slot(f"EUDR-{i+1:03d}", f"wf-{i+1:03d}")
        result = engine.acquire_slot("EUDR-EXTRA", "wf-extra")
        assert result is False

    def test_release_slot_success(self, parallel_execution_engine):
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        result = parallel_execution_engine.release_slot("EUDR-001", "wf-001")
        assert result is True

    def test_release_slot_decrements_count(self, parallel_execution_engine):
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        parallel_execution_engine.acquire_slot("EUDR-002", "wf-001")
        parallel_execution_engine.release_slot("EUDR-001", "wf-001")
        assert parallel_execution_engine.get_running_count() == 1

    def test_release_nonexistent_slot_returns_false(
        self, parallel_execution_engine,
    ):
        result = parallel_execution_engine.release_slot("EUDR-999", "wf-999")
        assert result is False

    def test_release_tracks_completion(self, parallel_execution_engine):
        """When completed=True, agent should be added to completed set."""
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        parallel_execution_engine.release_slot(
            "EUDR-001", "wf-001", completed=True,
        )
        assert "EUDR-001" in parallel_execution_engine._completed_agents["wf-001"]

    def test_release_not_completed_does_not_track(
        self, parallel_execution_engine,
    ):
        """When completed=False, agent should not be in completed set."""
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        parallel_execution_engine.release_slot(
            "EUDR-001", "wf-001", completed=False,
        )
        assert "EUDR-001" not in parallel_execution_engine._completed_agents.get(
            "wf-001", set(),
        )

    def test_release_frees_slot_for_reacquire(
        self, parallel_execution_engine,
    ):
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        parallel_execution_engine.release_slot("EUDR-001", "wf-001")
        result = parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        assert result is True


# =========================================================================
# Queue management
# =========================================================================


class TestQueueManagement:
    """Test execution queue enqueue/dequeue operations."""

    def test_enqueue_increases_queue_size(self, parallel_execution_engine):
        parallel_execution_engine.enqueue_agent("EUDR-001", "wf-001")
        assert parallel_execution_engine.get_queue_size() == 1

    def test_enqueue_multiple_agents(self, parallel_execution_engine):
        parallel_execution_engine.enqueue_agent("EUDR-001", "wf-001")
        parallel_execution_engine.enqueue_agent("EUDR-002", "wf-001")
        parallel_execution_engine.enqueue_agent("EUDR-003", "wf-001")
        assert parallel_execution_engine.get_queue_size() == 3

    def test_dequeue_returns_ready_agent(self, parallel_execution_engine):
        parallel_execution_engine.enqueue_agent("EUDR-001", "wf-001")
        agent = parallel_execution_engine.dequeue_next()
        assert agent is not None
        assert isinstance(agent, ReadyAgent)
        assert agent.agent_id == "EUDR-001"

    def test_dequeue_empty_returns_none(self, parallel_execution_engine):
        result = parallel_execution_engine.dequeue_next()
        assert result is None

    def test_dequeue_removes_from_queue(self, parallel_execution_engine):
        parallel_execution_engine.enqueue_agent("EUDR-001", "wf-001")
        parallel_execution_engine.dequeue_next()
        assert parallel_execution_engine.get_queue_size() == 0

    def test_enqueue_priority_ordering(self, parallel_execution_engine):
        """Higher priority (lower number) should dequeue first."""
        parallel_execution_engine.enqueue_agent(
            "EUDR-002", "wf-001", layer=0, priority=5,
        )
        parallel_execution_engine.enqueue_agent(
            "EUDR-001", "wf-001", layer=0, priority=1,
        )
        first = parallel_execution_engine.dequeue_next()
        assert first.agent_id == "EUDR-001"

    def test_enqueue_layer_ordering(self, parallel_execution_engine):
        """Lower layer should dequeue first regardless of priority."""
        parallel_execution_engine.enqueue_agent(
            "EUDR-016", "wf-001", layer=2, priority=1,
        )
        parallel_execution_engine.enqueue_agent(
            "EUDR-001", "wf-001", layer=0, priority=5,
        )
        first = parallel_execution_engine.dequeue_next()
        assert first.agent_id == "EUDR-001"

    def test_get_queue_for_workflow(self, parallel_execution_engine):
        parallel_execution_engine.enqueue_agent("EUDR-001", "wf-001")
        parallel_execution_engine.enqueue_agent("EUDR-002", "wf-001")
        parallel_execution_engine.enqueue_agent("EUDR-003", "wf-002")
        assert parallel_execution_engine.get_queue_for_workflow("wf-001") == 2
        assert parallel_execution_engine.get_queue_for_workflow("wf-002") == 1

    def test_get_queue_for_nonexistent_workflow(
        self, parallel_execution_engine,
    ):
        assert parallel_execution_engine.get_queue_for_workflow("wf-999") == 0

    def test_enqueue_with_custom_timeout(self, parallel_execution_engine):
        parallel_execution_engine.enqueue_agent(
            "EUDR-001", "wf-001", timeout_s=60,
        )
        agent = parallel_execution_engine.dequeue_next()
        assert agent.timeout_s == 60

    def test_enqueue_uses_config_timeout_when_none(
        self, parallel_execution_engine,
    ):
        """When timeout_s is not provided, uses config default."""
        parallel_execution_engine.enqueue_agent("EUDR-001", "wf-001")
        agent = parallel_execution_engine.dequeue_next()
        assert agent.timeout_s == parallel_execution_engine._config.agent_timeout_s


# =========================================================================
# Status tracking
# =========================================================================


class TestStatusTracking:
    """Test running count and agent listing."""

    def test_get_running_count_empty(self, parallel_execution_engine):
        assert parallel_execution_engine.get_running_count() == 0

    def test_get_running_count_after_acquires(
        self, parallel_execution_engine,
    ):
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        parallel_execution_engine.acquire_slot("EUDR-002", "wf-001")
        assert parallel_execution_engine.get_running_count() == 2

    def test_get_running_for_workflow(self, parallel_execution_engine):
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        parallel_execution_engine.acquire_slot("EUDR-002", "wf-001")
        parallel_execution_engine.acquire_slot("EUDR-003", "wf-002")
        assert parallel_execution_engine.get_running_for_workflow("wf-001") == 2
        assert parallel_execution_engine.get_running_for_workflow("wf-002") == 1

    def test_get_running_for_nonexistent_workflow(
        self, parallel_execution_engine,
    ):
        assert parallel_execution_engine.get_running_for_workflow("wf-999") == 0

    def test_get_running_agents(self, parallel_execution_engine):
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        parallel_execution_engine.acquire_slot("EUDR-002", "wf-001")
        agents = parallel_execution_engine.get_running_agents("wf-001")
        assert set(agents) == {"EUDR-001", "EUDR-002"}

    def test_get_running_agents_empty(self, parallel_execution_engine):
        agents = parallel_execution_engine.get_running_agents("wf-001")
        assert agents == []

    def test_get_timed_out_agents_none(self, parallel_execution_engine):
        parallel_execution_engine.acquire_slot(
            "EUDR-001", "wf-001", timeout_s=300,
        )
        timed_out = parallel_execution_engine.get_timed_out_agents()
        assert timed_out == []

    def test_get_timed_out_agents_detects_expired(
        self, parallel_execution_engine,
    ):
        parallel_execution_engine.acquire_slot(
            "EUDR-001", "wf-001", timeout_s=120,
        )
        # Backdate the slot to simulate timeout
        slot_key = "wf-001:EUDR-001"
        parallel_execution_engine._running_slots[slot_key].started_at = (
            datetime.now(timezone.utc) - timedelta(seconds=200)
        )
        timed_out = parallel_execution_engine.get_timed_out_agents()
        assert len(timed_out) == 1
        assert timed_out[0] == ("wf-001", "EUDR-001")

    def test_get_timed_out_mixed_results(self, parallel_execution_engine):
        """One timed out, one not."""
        parallel_execution_engine.acquire_slot(
            "EUDR-001", "wf-001", timeout_s=120,
        )
        parallel_execution_engine.acquire_slot(
            "EUDR-002", "wf-001", timeout_s=300,
        )
        # Only EUDR-001 timed out
        slot_key = "wf-001:EUDR-001"
        parallel_execution_engine._running_slots[slot_key].started_at = (
            datetime.now(timezone.utc) - timedelta(seconds=200)
        )
        timed_out = parallel_execution_engine.get_timed_out_agents()
        assert len(timed_out) == 1
        agent_ids = [t[1] for t in timed_out]
        assert "EUDR-001" in agent_ids
        assert "EUDR-002" not in agent_ids


# =========================================================================
# Circuit breaker integration
# =========================================================================


class TestCircuitBreakerIntegration:
    """Test circuit breaker state management."""

    def test_default_state_is_closed(self, parallel_execution_engine):
        state = parallel_execution_engine.get_circuit_breaker_state("EUDR-001")
        assert state == CircuitBreakerState.CLOSED

    def test_set_state_open(self, parallel_execution_engine):
        parallel_execution_engine.set_circuit_breaker_state(
            "EUDR-003", CircuitBreakerState.OPEN,
        )
        state = parallel_execution_engine.get_circuit_breaker_state("EUDR-003")
        assert state == CircuitBreakerState.OPEN

    def test_set_state_half_open(self, parallel_execution_engine):
        parallel_execution_engine.set_circuit_breaker_state(
            "EUDR-003", CircuitBreakerState.HALF_OPEN,
        )
        state = parallel_execution_engine.get_circuit_breaker_state("EUDR-003")
        assert state == CircuitBreakerState.HALF_OPEN

    def test_set_state_back_to_closed(self, parallel_execution_engine):
        parallel_execution_engine.set_circuit_breaker_state(
            "EUDR-003", CircuitBreakerState.OPEN,
        )
        parallel_execution_engine.set_circuit_breaker_state(
            "EUDR-003", CircuitBreakerState.CLOSED,
        )
        state = parallel_execution_engine.get_circuit_breaker_state("EUDR-003")
        assert state == CircuitBreakerState.CLOSED

    def test_open_circuit_excludes_from_ready_agents(
        self, parallel_execution_engine,
    ):
        """Integration: open circuit breaker prevents readiness."""
        defn = _simple_definition(
            nodes_spec=[("EUDR-003", 0), ("EUDR-004", 0)],
            edges_spec=[],
        )
        parallel_execution_engine.set_circuit_breaker_state(
            "EUDR-003", CircuitBreakerState.OPEN,
        )
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=defn,
            completed=set(),
            running=set(),
        )
        assert "EUDR-003" not in ready
        assert "EUDR-004" in ready

    def test_multiple_agents_independent_circuit_states(
        self, parallel_execution_engine,
    ):
        """Each agent has its own independent circuit breaker state."""
        parallel_execution_engine.set_circuit_breaker_state(
            "EUDR-001", CircuitBreakerState.OPEN,
        )
        parallel_execution_engine.set_circuit_breaker_state(
            "EUDR-002", CircuitBreakerState.HALF_OPEN,
        )
        assert (
            parallel_execution_engine.get_circuit_breaker_state("EUDR-001")
            == CircuitBreakerState.OPEN
        )
        assert (
            parallel_execution_engine.get_circuit_breaker_state("EUDR-002")
            == CircuitBreakerState.HALF_OPEN
        )
        assert (
            parallel_execution_engine.get_circuit_breaker_state("EUDR-003")
            == CircuitBreakerState.CLOSED
        )


# =========================================================================
# ETA estimation
# =========================================================================


class TestETAEstimation:
    """Test workflow completion ETA estimation."""

    def test_eta_zero_remaining(self, parallel_execution_engine):
        eta = parallel_execution_engine.estimate_eta(
            workflow_id="wf-001",
            total_agents=10,
            completed_count=10,
        )
        assert eta == 0

    def test_eta_all_remaining(self, parallel_execution_engine):
        """10 agents, 0 completed, 30s avg, max_concurrent=10 -> 30s."""
        eta = parallel_execution_engine.estimate_eta(
            workflow_id="wf-001",
            total_agents=10,
            completed_count=0,
            avg_duration_s=30.0,
        )
        # parallelism = min(10, 10) = 10
        # eta = 10 * 30 / 10 = 30
        assert eta == 30

    def test_eta_partial_completion(self, parallel_execution_engine):
        """25 agents, 15 completed, 30s avg, max_concurrent=10 -> 10*30/10=30."""
        eta = parallel_execution_engine.estimate_eta(
            workflow_id="wf-001",
            total_agents=25,
            completed_count=15,
            avg_duration_s=30.0,
        )
        remaining = 10
        parallelism = min(10, remaining)
        expected = int(remaining * 30.0 / parallelism)
        assert eta == expected

    def test_eta_remaining_less_than_parallelism(self, parallel_execution_engine):
        """3 remaining, max_concurrent=10 -> parallelism=3."""
        eta = parallel_execution_engine.estimate_eta(
            workflow_id="wf-001",
            total_agents=25,
            completed_count=22,
            avg_duration_s=30.0,
        )
        remaining = 3
        parallelism = min(10, remaining)
        expected = int(remaining * 30.0 / parallelism)
        assert eta == expected  # 3*30/3 = 30

    def test_eta_custom_avg_duration(self, parallel_execution_engine):
        eta = parallel_execution_engine.estimate_eta(
            workflow_id="wf-001",
            total_agents=10,
            completed_count=0,
            avg_duration_s=60.0,
        )
        # 10 * 60 / 10 = 60
        assert eta == 60

    def test_eta_negative_remaining_returns_zero(
        self, parallel_execution_engine,
    ):
        """completed > total should return 0."""
        eta = parallel_execution_engine.estimate_eta(
            workflow_id="wf-001",
            total_agents=5,
            completed_count=10,
        )
        assert eta == 0

    @pytest.mark.parametrize("total,completed,avg,max_conc,expected", [
        (20, 10, 30.0, 10, 30),   # 10 remaining, min(10,10)=10, 10*30/10=30
        (20, 10, 30.0, 5, 60),    # 10 remaining, min(5,10)=5, 10*30/5=60
        (20, 19, 30.0, 10, 30),   # 1 remaining, min(10,1)=1, 1*30/1=30
        (20, 0, 10.0, 10, 20),    # 20 remaining, min(10,20)=10, 20*10/10=20
        (1, 0, 100.0, 10, 100),   # 1 remaining, min(10,1)=1, 1*100/1=100
    ])
    def test_eta_parametrized_scenarios(
        self, default_config, total, completed, avg, max_conc, expected,
    ):
        cfg = DueDiligenceOrchestratorConfig(max_concurrent_agents=max_conc)
        set_config(cfg)
        engine = ParallelExecutionEngine(config=cfg)
        eta = engine.estimate_eta(
            workflow_id="wf-001",
            total_agents=total,
            completed_count=completed,
            avg_duration_s=avg,
        )
        assert eta == expected


# =========================================================================
# Workflow cleanup
# =========================================================================


class TestWorkflowCleanup:
    """Test workflow cleanup operations."""

    def test_cleanup_removes_running_slots(self, parallel_execution_engine):
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        parallel_execution_engine.acquire_slot("EUDR-002", "wf-001")
        cleaned = parallel_execution_engine.cleanup_workflow("wf-001")
        assert cleaned == 2
        assert parallel_execution_engine.get_running_count() == 0

    def test_cleanup_removes_queue_entries(self, parallel_execution_engine):
        parallel_execution_engine.enqueue_agent("EUDR-001", "wf-001")
        parallel_execution_engine.enqueue_agent("EUDR-002", "wf-001")
        cleaned = parallel_execution_engine.cleanup_workflow("wf-001")
        assert cleaned == 2
        assert parallel_execution_engine.get_queue_size() == 0

    def test_cleanup_removes_completed_tracking(
        self, parallel_execution_engine,
    ):
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        parallel_execution_engine.release_slot(
            "EUDR-001", "wf-001", completed=True,
        )
        parallel_execution_engine.cleanup_workflow("wf-001")
        assert "wf-001" not in parallel_execution_engine._completed_agents

    def test_cleanup_only_affects_target_workflow(
        self, parallel_execution_engine,
    ):
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        parallel_execution_engine.acquire_slot("EUDR-002", "wf-002")
        parallel_execution_engine.enqueue_agent("EUDR-003", "wf-001")
        parallel_execution_engine.enqueue_agent("EUDR-004", "wf-002")
        parallel_execution_engine.cleanup_workflow("wf-001")
        # wf-002 should be untouched
        assert parallel_execution_engine.get_running_for_workflow("wf-002") == 1
        assert parallel_execution_engine.get_queue_for_workflow("wf-002") == 1
        # wf-001 should be gone
        assert parallel_execution_engine.get_running_for_workflow("wf-001") == 0
        assert parallel_execution_engine.get_queue_for_workflow("wf-001") == 0

    def test_cleanup_nonexistent_workflow_returns_zero(
        self, parallel_execution_engine,
    ):
        cleaned = parallel_execution_engine.cleanup_workflow("wf-nonexistent")
        assert cleaned == 0

    def test_cleanup_combined_slots_and_queue(
        self, parallel_execution_engine,
    ):
        """Count includes both slots and queue entries."""
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        parallel_execution_engine.enqueue_agent("EUDR-002", "wf-001")
        parallel_execution_engine.enqueue_agent("EUDR-003", "wf-001")
        cleaned = parallel_execution_engine.cleanup_workflow("wf-001")
        assert cleaned == 3  # 1 slot + 2 queue entries

    def test_cleanup_idempotent(self, parallel_execution_engine):
        """Calling cleanup twice does not raise errors."""
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        parallel_execution_engine.cleanup_workflow("wf-001")
        cleaned = parallel_execution_engine.cleanup_workflow("wf-001")
        assert cleaned == 0


# =========================================================================
# Thread safety
# =========================================================================


class TestThreadSafety:
    """Test thread-safe operations under concurrent access."""

    def test_concurrent_acquire_respects_global_limit(self, default_config):
        """Multiple threads acquiring slots should respect global limit."""
        cfg = DueDiligenceOrchestratorConfig(
            max_concurrent_agents=10,
            global_concurrency_limit=10,
        )
        set_config(cfg)
        engine = ParallelExecutionEngine(config=cfg)

        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(20)

        def acquire(idx):
            barrier.wait()
            result = engine.acquire_slot(
                agent_id=f"AGENT-{idx:03d}",
                workflow_id=f"wf-{idx:03d}",
            )
            with lock:
                results.append(result)

        threads = [threading.Thread(target=acquire, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        success_count = sum(1 for r in results if r is True)
        assert success_count == 10
        assert engine.get_running_count() == 10

    def test_concurrent_enqueue_dequeue(self, parallel_execution_engine):
        """Concurrent enqueue and dequeue should not corrupt state."""
        barrier = threading.Barrier(10)
        dequeued = []

        def enqueue_work(idx):
            barrier.wait()
            parallel_execution_engine.enqueue_agent(
                f"AGENT-{idx:03d}", "wf-001", layer=0, priority=idx,
            )

        def dequeue_work():
            time.sleep(0.05)  # Give enqueue threads a head start
            for _ in range(5):
                result = parallel_execution_engine.dequeue_next()
                if result:
                    dequeued.append(result.agent_id)

        enqueue_threads = [
            threading.Thread(target=enqueue_work, args=(i,)) for i in range(10)
        ]
        dequeue_threads = [
            threading.Thread(target=dequeue_work) for _ in range(2)
        ]
        for t in enqueue_threads:
            t.start()
        for t in dequeue_threads:
            t.start()
        for t in enqueue_threads + dequeue_threads:
            t.join()

        # All dequeued + remaining in queue should equal 10
        remaining = parallel_execution_engine.get_queue_size()
        assert len(dequeued) + remaining == 10


# =========================================================================
# Integration with standard EUDR workflow
# =========================================================================


class TestStandardWorkflowIntegration:
    """Integration tests using the full standard EUDR workflow definition."""

    def test_phase1_agents_become_ready_after_root(
        self, parallel_execution_engine, standard_workflow_definition,
    ):
        """After EUDR-001 completes, its dependents should become ready."""
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=standard_workflow_definition,
            completed={"EUDR-001"},
            running=set(),
        )
        # EUDR-002, EUDR-006, EUDR-007, EUDR-008 depend on EUDR-001
        for expected in ["EUDR-002", "EUDR-006", "EUDR-007", "EUDR-008"]:
            assert expected in ready, f"{expected} should be ready after EUDR-001"

    def test_quality_gate_blocked_until_deps_complete(
        self, parallel_execution_engine, standard_workflow_definition,
    ):
        """QG-1 should not be ready until all its dependencies complete."""
        # Complete only some phase 1 agents
        ready = parallel_execution_engine.get_ready_agents(
            workflow_id="wf-001",
            definition=standard_workflow_definition,
            completed={"EUDR-001", "EUDR-002", "EUDR-003"},
            running=set(),
        )
        assert "QG-1" not in ready

    def test_acquire_release_full_lifecycle(
        self, parallel_execution_engine,
    ):
        """Full lifecycle: acquire -> run -> release -> reacquire."""
        assert parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        assert parallel_execution_engine.get_running_count() == 1

        assert parallel_execution_engine.release_slot(
            "EUDR-001", "wf-001", completed=True,
        )
        assert parallel_execution_engine.get_running_count() == 0
        assert "EUDR-001" in parallel_execution_engine._completed_agents["wf-001"]

    def test_full_workflow_cleanup_after_completion(
        self, parallel_execution_engine,
    ):
        """After workflow finishes, cleanup removes all tracking state."""
        parallel_execution_engine.acquire_slot("EUDR-001", "wf-001")
        parallel_execution_engine.release_slot(
            "EUDR-001", "wf-001", completed=True,
        )
        parallel_execution_engine.enqueue_agent("EUDR-002", "wf-001")

        cleaned = parallel_execution_engine.cleanup_workflow("wf-001")
        assert cleaned == 1  # 1 queue entry (slot already released)
        assert parallel_execution_engine.get_running_for_workflow("wf-001") == 0
        assert parallel_execution_engine.get_queue_for_workflow("wf-001") == 0
        assert "wf-001" not in parallel_execution_engine._completed_agents
