# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Orchestrator Unit Tests (AGENT-FOUND-001)
==============================================================

Provides shared DAG fixtures, mock agents, retry/timeout policies,
and helper utilities for testing the DAG Execution Engine.

All tests are self-contained with no external dependencies.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest


# ---------------------------------------------------------------------------
# Async helper for Windows compatibility
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine synchronously. Windows-compatible."""
    if sys.platform == "win32":
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    else:
        return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Enums (mirror expected module enums)
# ---------------------------------------------------------------------------


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class OnFailure(str, Enum):
    STOP = "stop"
    SKIP = "skip"
    COMPENSATE = "compensate"


class RetryStrategy(str, Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"
    FIBONACCI = "fibonacci"


class OnTimeout(str, Enum):
    FAIL = "fail"
    SKIP = "skip"
    COMPENSATE = "compensate"


# ---------------------------------------------------------------------------
# Lightweight model stubs for test fixtures
# ---------------------------------------------------------------------------


@dataclass
class RetryPolicyData:
    max_retries: int = 3
    strategy: str = "exponential"
    base_delay: float = 0.01
    max_delay: float = 1.0
    jitter: bool = False
    retryable_exceptions: List[str] = field(default_factory=lambda: ["Exception"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "strategy": self.strategy,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "jitter": self.jitter,
            "retryable_exceptions": self.retryable_exceptions,
        }


@dataclass
class TimeoutPolicyData:
    timeout_seconds: float = 5.0
    on_timeout: str = "fail"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "on_timeout": self.on_timeout,
        }


@dataclass
class DAGNodeData:
    node_id: str
    agent_id: str
    depends_on: List[str] = field(default_factory=list)
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_key: str = ""
    condition: Optional[str] = None
    retry_policy: Optional[RetryPolicyData] = None
    timeout_policy: Optional[TimeoutPolicyData] = None
    on_failure: str = "stop"
    compensation_handler: Optional[str] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "node_id": self.node_id,
            "agent_id": self.agent_id,
            "depends_on": self.depends_on,
            "input_mapping": self.input_mapping,
            "output_key": self.output_key or self.node_id,
            "on_failure": self.on_failure,
            "priority": self.priority,
            "metadata": self.metadata,
        }
        if self.condition:
            d["condition"] = self.condition
        if self.retry_policy:
            d["retry_policy"] = self.retry_policy.to_dict()
        if self.timeout_policy:
            d["timeout_policy"] = self.timeout_policy.to_dict()
        if self.compensation_handler:
            d["compensation_handler"] = self.compensation_handler
        return d


@dataclass
class DAGWorkflowData:
    dag_id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    nodes: Dict[str, DAGNodeData] = field(default_factory=dict)
    default_retry_policy: Optional[RetryPolicyData] = None
    default_timeout_policy: Optional[TimeoutPolicyData] = None
    on_failure: str = "fail_fast"
    max_parallel_nodes: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dag_id": self.dag_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "default_retry_policy": self.default_retry_policy.to_dict()
            if self.default_retry_policy
            else None,
            "default_timeout_policy": self.default_timeout_policy.to_dict()
            if self.default_timeout_policy
            else None,
            "on_failure": self.on_failure,
            "max_parallel_nodes": self.max_parallel_nodes,
            "metadata": self.metadata,
        }

    def get_adjacency(self) -> Dict[str, List[str]]:
        """Return adjacency list: node_id -> list of dependents."""
        adj: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            for dep in node.depends_on:
                if dep in adj:
                    adj[dep].append(nid)
        return adj


# ---------------------------------------------------------------------------
# DAG Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_linear_dag() -> DAGWorkflowData:
    """3 nodes: A -> B -> C (linear chain)."""
    return DAGWorkflowData(
        dag_id="linear-dag",
        name="Linear DAG",
        description="Simple linear pipeline",
        nodes={
            "A": DAGNodeData(node_id="A", agent_id="agent_a", depends_on=[]),
            "B": DAGNodeData(node_id="B", agent_id="agent_b", depends_on=["A"]),
            "C": DAGNodeData(node_id="C", agent_id="agent_c", depends_on=["B"]),
        },
    )


@pytest.fixture
def sample_diamond_dag() -> DAGWorkflowData:
    """4 nodes: A -> B, A -> C, B -> D, C -> D (diamond pattern)."""
    return DAGWorkflowData(
        dag_id="diamond-dag",
        name="Diamond DAG",
        description="Diamond dependency pattern",
        nodes={
            "A": DAGNodeData(node_id="A", agent_id="agent_a", depends_on=[]),
            "B": DAGNodeData(node_id="B", agent_id="agent_b", depends_on=["A"]),
            "C": DAGNodeData(node_id="C", agent_id="agent_c", depends_on=["A"]),
            "D": DAGNodeData(node_id="D", agent_id="agent_d", depends_on=["B", "C"]),
        },
    )


@pytest.fixture
def sample_complex_dag() -> DAGWorkflowData:
    """6 nodes: intake -> validate -> scope1/scope2 -> aggregate -> report."""
    return DAGWorkflowData(
        dag_id="emissions-calculation",
        name="Emissions Calculation",
        description="Calculate Scope 1+2 emissions",
        nodes={
            "intake": DAGNodeData(
                node_id="intake",
                agent_id="intake_agent",
                depends_on=[],
                output_key="raw_data",
            ),
            "validate": DAGNodeData(
                node_id="validate",
                agent_id="validation_agent",
                depends_on=["intake"],
                input_mapping={"data": "results.intake.raw_data"},
                output_key="validated_data",
            ),
            "scope1": DAGNodeData(
                node_id="scope1",
                agent_id="scope1_calc_agent",
                depends_on=["validate"],
                retry_policy=RetryPolicyData(max_retries=3, base_delay=0.01),
            ),
            "scope2": DAGNodeData(
                node_id="scope2",
                agent_id="scope2_calc_agent",
                depends_on=["validate"],
                retry_policy=RetryPolicyData(max_retries=3, base_delay=0.01),
            ),
            "aggregate": DAGNodeData(
                node_id="aggregate",
                agent_id="aggregation_agent",
                depends_on=["scope1", "scope2"],
                output_key="total_emissions",
            ),
            "report": DAGNodeData(
                node_id="report",
                agent_id="reporting_agent",
                depends_on=["aggregate"],
            ),
        },
        default_retry_policy=RetryPolicyData(max_retries=2, base_delay=0.01),
        default_timeout_policy=TimeoutPolicyData(timeout_seconds=5.0),
        max_parallel_nodes=10,
    )


@pytest.fixture
def sample_wide_dag() -> DAGWorkflowData:
    """1 root -> 5 parallel -> 1 sink (wide parallel pattern)."""
    nodes = {
        "root": DAGNodeData(node_id="root", agent_id="root_agent", depends_on=[]),
    }
    for i in range(5):
        nid = f"parallel_{i}"
        nodes[nid] = DAGNodeData(
            node_id=nid, agent_id=f"parallel_agent_{i}", depends_on=["root"]
        )
    nodes["sink"] = DAGNodeData(
        node_id="sink",
        agent_id="sink_agent",
        depends_on=[f"parallel_{i}" for i in range(5)],
    )
    return DAGWorkflowData(
        dag_id="wide-dag",
        name="Wide Parallel DAG",
        description="Root -> 5 parallel -> Sink",
        nodes=nodes,
    )


@pytest.fixture
def sample_invalid_cycle_dag() -> DAGWorkflowData:
    """Invalid DAG: A -> B -> C -> A (cycle)."""
    return DAGWorkflowData(
        dag_id="cycle-dag",
        name="Invalid Cycle DAG",
        nodes={
            "A": DAGNodeData(node_id="A", agent_id="agent_a", depends_on=["C"]),
            "B": DAGNodeData(node_id="B", agent_id="agent_b", depends_on=["A"]),
            "C": DAGNodeData(node_id="C", agent_id="agent_c", depends_on=["B"]),
        },
    )


@pytest.fixture
def sample_invalid_self_dep_dag() -> DAGWorkflowData:
    """Invalid DAG: A depends on itself."""
    return DAGWorkflowData(
        dag_id="self-dep-dag",
        name="Invalid Self-Dependency DAG",
        nodes={
            "A": DAGNodeData(node_id="A", agent_id="agent_a", depends_on=["A"]),
        },
    )


@pytest.fixture
def sample_invalid_missing_dep_dag() -> DAGWorkflowData:
    """Invalid DAG: A depends on nonexistent node B."""
    return DAGWorkflowData(
        dag_id="missing-dep-dag",
        name="Invalid Missing Dependency DAG",
        nodes={
            "A": DAGNodeData(
                node_id="A", agent_id="agent_a", depends_on=["nonexistent"]
            ),
        },
    )


# ---------------------------------------------------------------------------
# Policy Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_retry_policy() -> RetryPolicyData:
    """Fast retry policy for testing: max_retries=3, exponential, base_delay=0.01."""
    return RetryPolicyData(
        max_retries=3,
        strategy="exponential",
        base_delay=0.01,
        max_delay=0.1,
        jitter=False,
    )


@pytest.fixture
def sample_timeout_policy() -> TimeoutPolicyData:
    """Timeout policy for testing: 5 seconds."""
    return TimeoutPolicyData(timeout_seconds=5.0, on_timeout="fail")


# ---------------------------------------------------------------------------
# Mock Agent Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sync_agent():
    """Mock agent with .run() that returns deterministic results."""

    def _make_agent(
        result: Optional[Dict[str, Any]] = None, agent_id: str = "mock_agent"
    ):
        agent = MagicMock()
        agent.agent_id = agent_id
        agent.run.return_value = result or {
            "success": True,
            "data": {"output": f"result_from_{agent_id}"},
        }
        # Explicitly set run_async to None so MagicMock does not auto-create it
        agent.run_async = None
        return agent

    return _make_agent


@pytest.fixture
def mock_async_agent():
    """Mock agent with .run_async() that returns deterministic results."""

    def _make_agent(
        result: Optional[Dict[str, Any]] = None, agent_id: str = "mock_async_agent"
    ):
        agent = MagicMock()
        agent.agent_id = agent_id
        agent.run = None  # No sync run
        agent.run_async = AsyncMock(
            return_value=result
            or {"success": True, "data": {"output": f"result_from_{agent_id}"}}
        )
        return agent

    return _make_agent


@pytest.fixture
def mock_failing_agent():
    """Mock agent that fails N times then succeeds."""

    def _make_agent(fail_count: int = 2, agent_id: str = "failing_agent"):
        agent = MagicMock()
        agent.agent_id = agent_id
        # Explicitly set run_async to None so MagicMock does not auto-create it
        agent.run_async = None
        call_count = {"n": 0}

        def _run(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= fail_count:
                raise RuntimeError(
                    f"Simulated failure #{call_count['n']} from {agent_id}"
                )
            return {"success": True, "data": {"output": f"result_from_{agent_id}"}}

        agent.run.side_effect = _run
        return agent

    return _make_agent


@pytest.fixture
def mock_timeout_agent():
    """Mock agent that sleeps longer than timeout."""

    def _make_agent(sleep_seconds: float = 10.0, agent_id: str = "timeout_agent"):
        agent = MagicMock()
        agent.agent_id = agent_id

        async def _run_async(*args, **kwargs):
            await asyncio.sleep(sleep_seconds)
            return {"success": True, "data": {"output": "should_not_reach"}}

        agent.run_async = _run_async
        agent.run = None
        return agent

    return _make_agent


# ---------------------------------------------------------------------------
# Agent Registry Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent_registry(mock_sync_agent):
    """Mock agent registry that returns mock agents for any agent_id."""
    registry = {}

    def _get_agent(agent_id: str):
        if agent_id not in registry:
            registry[agent_id] = mock_sync_agent(agent_id=agent_id)
        return registry[agent_id]

    return _get_agent


# ---------------------------------------------------------------------------
# Config Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Test configuration with short timeouts for fast tests."""
    return {
        "service_name": "test-orchestrator",
        "environment": "test",
        "default_timeout_seconds": 5.0,
        "default_max_retries": 2,
        "default_retry_strategy": "exponential",
        "default_base_delay": 0.01,
        "max_parallel_nodes": 10,
        "checkpoint_strategy": "memory",
        "checkpoint_dir": "/tmp/test-checkpoints",
        "deterministic_mode": True,
        "log_level": "DEBUG",
        "metrics_enabled": False,
    }


# ---------------------------------------------------------------------------
# Utility Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def compute_hash():
    """Utility to compute SHA-256 hash of data."""

    def _hash(data: Any) -> str:
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    return _hash


@pytest.fixture
def run_async():
    """Provide _run_async helper as a fixture."""
    return _run_async


# ---------------------------------------------------------------------------
# Override parent conftest fixtures that may interfere
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest mock_agents (not needed for orchestrator tests)."""
    yield
