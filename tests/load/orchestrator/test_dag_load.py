# -*- coding: utf-8 -*-
"""
Load Tests for DAG Orchestrator (AGENT-FOUND-001)

Tests performance under load: concurrent executions, large DAG validation,
checkpoint throughput, and API latency.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import sys
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

# Capture original socket before NetworkBlocker
_ORIGINAL_SOCKET = socket.socket
_ORIGINAL_CREATE_CONNECTION = socket.create_connection


# ---------------------------------------------------------------------------
# Restore sockets and override parent fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_sockets():
    saved = socket.socket
    saved_cc = socket.create_connection
    socket.socket = _ORIGINAL_SOCKET
    socket.create_connection = _ORIGINAL_CREATE_CONNECTION
    yield
    socket.socket = saved
    socket.create_connection = saved_cc


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest mock_agents."""
    yield


# ---------------------------------------------------------------------------
# Imports from unit test inline implementations
# ---------------------------------------------------------------------------

from tests.unit.orchestrator.test_dag_executor import (
    CheckpointStore,
    DAGExecutor,
)
from tests.unit.orchestrator.test_dag_validator import validate_dag
from tests.unit.orchestrator.test_topological_sort import (
    topological_sort,
    level_grouping,
)
from tests.unit.orchestrator.test_checkpoint_store import (
    DAGCheckpointData,
    MemoryDAGCheckpointStore,
)
from tests.unit.orchestrator.conftest import DAGNodeData, DAGWorkflowData


def _make_registry():
    def _get(agent_id):
        mock = MagicMock()
        mock.run.return_value = {"result": f"from_{agent_id}"}
        mock.run_async = None
        return mock
    return _get


def _generate_linear_dag(size: int) -> DAGWorkflowData:
    """Generate a linear DAG with N nodes."""
    nodes = {}
    for i in range(size):
        nid = f"node_{i:04d}"
        deps = [f"node_{i-1:04d}"] if i > 0 else []
        nodes[nid] = DAGNodeData(node_id=nid, agent_id=f"agent_{i}", depends_on=deps)
    return DAGWorkflowData(dag_id=f"linear-{size}", name=f"Linear {size}", nodes=nodes)


def _generate_wide_dag(parallel_count: int) -> DAGWorkflowData:
    """Generate a wide DAG: root -> N parallel -> sink."""
    nodes = {"root": DAGNodeData(node_id="root", agent_id="root_agent")}
    parallel_ids = []
    for i in range(parallel_count):
        nid = f"parallel_{i:04d}"
        nodes[nid] = DAGNodeData(
            node_id=nid, agent_id=f"parallel_{i}", depends_on=["root"]
        )
        parallel_ids.append(nid)
    nodes["sink"] = DAGNodeData(
        node_id="sink", agent_id="sink_agent", depends_on=parallel_ids
    )
    return DAGWorkflowData(
        dag_id=f"wide-{parallel_count}", name=f"Wide {parallel_count}", nodes=nodes
    )


def _generate_diamond_layers(layers: int, width: int) -> DAGWorkflowData:
    """Generate a multi-layer diamond DAG."""
    nodes = {}
    prev_layer = []

    for layer in range(layers):
        current_layer = []
        for w in range(width if layer > 0 and layer < layers - 1 else 1):
            nid = f"L{layer}_N{w}"
            deps = prev_layer if prev_layer else []
            nodes[nid] = DAGNodeData(
                node_id=nid, agent_id=f"agent_{layer}_{w}", depends_on=deps
            )
            current_layer.append(nid)
        prev_layer = current_layer

    return DAGWorkflowData(
        dag_id=f"diamond-{layers}x{width}",
        name=f"Diamond {layers}x{width}",
        nodes=nodes,
    )


# ===========================================================================
# Test Classes
# ===========================================================================


@pytest.mark.performance
class TestConcurrentDAGExecutions:
    """Test 50 concurrent DAG executions."""

    def test_50_concurrent_dag_executions(self):
        """50 concurrent DAG executions complete within 30 seconds."""
        dag = DAGWorkflowData(
            dag_id="load-dag",
            name="Load DAG",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A"]),
                "C": DAGNodeData(node_id="C", agent_id="c", depends_on=["A"]),
                "D": DAGNodeData(
                    node_id="D", agent_id="d", depends_on=["B", "C"]
                ),
            },
        )

        async def _run_all():
            tasks = []
            for i in range(50):
                executor = DAGExecutor(
                    agent_registry=_make_registry(),
                    checkpoint_store=CheckpointStore(),
                )
                tasks.append(executor.execute(dag, execution_id=f"load-{i}"))
            return await asyncio.gather(*tasks)

        start = time.monotonic()
        if sys.platform == "win32":
            loop = asyncio.new_event_loop()
            try:
                results = loop.run_until_complete(_run_all())
            finally:
                loop.close()
        else:
            results = asyncio.get_event_loop().run_until_complete(_run_all())
        elapsed = time.monotonic() - start

        assert len(results) == 50
        completed = sum(1 for r in results if r.status == "completed")
        assert completed == 50, f"Only {completed}/50 completed"
        assert elapsed < 30.0, f"50 concurrent executions took {elapsed:.2f}s"


@pytest.mark.performance
class TestLargeDAGValidation:
    """Test validation performance for large DAGs."""

    def test_100_node_linear_dag_validation(self):
        """100-node linear DAG validates in under 1 second."""
        dag = _generate_linear_dag(100)
        start = time.monotonic()
        result = validate_dag(dag)
        elapsed = time.monotonic() - start

        assert result.is_valid
        assert elapsed < 1.0, f"100-node validation took {elapsed:.4f}s"

    def test_500_node_linear_dag_validation(self):
        """500-node linear DAG validates in under 5 seconds."""
        dag = _generate_linear_dag(500)
        start = time.monotonic()
        result = validate_dag(dag)
        elapsed = time.monotonic() - start

        assert result.is_valid
        assert elapsed < 5.0, f"500-node validation took {elapsed:.4f}s"

    def test_500_node_topological_sort(self):
        """500-node DAG sorts in under 5 seconds."""
        dag = _generate_linear_dag(500)
        start = time.monotonic()
        order = topological_sort(dag)
        elapsed = time.monotonic() - start

        assert len(order) == 500
        assert elapsed < 5.0, f"500-node sort took {elapsed:.4f}s"

    def test_200_parallel_wide_dag_validation(self):
        """Wide DAG with 200 parallel nodes validates quickly."""
        dag = _generate_wide_dag(200)
        start = time.monotonic()
        result = validate_dag(dag)
        elapsed = time.monotonic() - start

        assert result.is_valid
        assert elapsed < 2.0, f"200-parallel validation took {elapsed:.4f}s"

    def test_500_node_level_grouping(self):
        """500-node level grouping completes under 5 seconds."""
        dag = _generate_linear_dag(500)
        start = time.monotonic()
        levels = level_grouping(dag)
        elapsed = time.monotonic() - start

        assert len(levels) == 500  # Linear = 500 levels
        assert elapsed < 5.0, f"500-node level grouping took {elapsed:.4f}s"


@pytest.mark.performance
class TestCheckpointThroughput:
    """Test checkpoint store throughput."""

    def test_1000_checkpoint_saves(self):
        """1000 checkpoint saves in under 5 seconds."""
        store = MemoryDAGCheckpointStore()

        start = time.monotonic()
        for i in range(1000):
            cp = DAGCheckpointData(
                node_id=f"node_{i}",
                status="completed",
                output={"value": i},
                output_hash=hashlib.sha256(str(i).encode()).hexdigest(),
            )
            store.save("load-exec", cp)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"1000 saves took {elapsed:.4f}s"
        # Guard against zero elapsed on fast machines
        if elapsed > 0:
            throughput = 1000 / elapsed
            assert throughput > 200, f"Throughput: {throughput:.0f} saves/sec"

    def test_1000_checkpoint_loads(self):
        """1000 checkpoint loads in under 5 seconds."""
        store = MemoryDAGCheckpointStore()
        for i in range(1000):
            store.save(
                "load-exec",
                DAGCheckpointData(node_id=f"node_{i}", status="completed"),
            )

        start = time.monotonic()
        for i in range(1000):
            store.load("load-exec", f"node_{i}")
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"1000 loads took {elapsed:.4f}s"

    def test_get_completed_nodes_at_scale(self):
        """Get completed nodes from 1000 checkpoints."""
        store = MemoryDAGCheckpointStore()
        for i in range(1000):
            status = "completed" if i % 2 == 0 else "failed"
            store.save(
                "scale-exec",
                DAGCheckpointData(node_id=f"node_{i}", status=status),
            )

        start = time.monotonic()
        completed = store.get_completed_nodes("scale-exec")
        elapsed = time.monotonic() - start

        assert len(completed) == 500
        assert elapsed < 1.0, f"Get completed took {elapsed:.4f}s"


@pytest.mark.performance
class TestAPILatencyUnderLoad:
    """Test API handler latency under simulated load."""

    def test_create_dag_latency(self):
        """Creating 100 DAGs should complete in under 5 seconds."""
        from tests.integration.orchestrator.test_api_integration import OrchestratorAPI

        api = OrchestratorAPI()
        start = time.monotonic()
        for i in range(100):
            api.create_dag({
                "dag_id": f"load-dag-{i}",
                "name": f"Load DAG {i}",
                "nodes": {
                    "A": {"agent_id": "a", "depends_on": []},
                    "B": {"agent_id": "b", "depends_on": ["A"]},
                },
            })
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"100 creates took {elapsed:.4f}s"

    def test_list_dags_latency(self):
        """Listing 100 DAGs should be under 1 second."""
        from tests.integration.orchestrator.test_api_integration import OrchestratorAPI

        api = OrchestratorAPI()
        for i in range(100):
            api.create_dag({
                "dag_id": f"list-dag-{i}",
                "name": f"List DAG {i}",
                "nodes": {"A": {"agent_id": "a", "depends_on": []}},
            })

        start = time.monotonic()
        for _ in range(100):
            api.list_dags()
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"100 list operations took {elapsed:.4f}s"
