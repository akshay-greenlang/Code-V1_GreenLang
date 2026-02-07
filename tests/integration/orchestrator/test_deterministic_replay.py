# -*- coding: utf-8 -*-
"""
Deterministic Replay Integration Tests (AGENT-FOUND-001)

Tests that executing the same DAG with the same inputs produces
identical execution traces and provenance chains.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock

import pytest

from tests.integration.orchestrator.conftest import _run_async
from tests.unit.orchestrator.test_dag_executor import (
    CheckpointStore,
    DAGExecutor,
)
from tests.unit.orchestrator.conftest import DAGNodeData, DAGWorkflowData


def _make_deterministic_registry():
    """Registry that always returns the same output for a given agent_id."""

    def _get(agent_id):
        mock = MagicMock()
        mock.run_async = None
        # Deterministic: output depends only on agent_id
        mock.run.return_value = {
            "agent": agent_id,
            "value": hash(agent_id) % 1000,
        }
        return mock

    return _get


@pytest.mark.integration
class TestIdenticalTracesForSameInputs:
    """Test that same DAG + inputs produce identical execution results."""

    def test_two_executions_same_topology(self):
        dag = DAGWorkflowData(
            dag_id="replay-dag",
            name="Replay DAG",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A"]),
                "C": DAGNodeData(node_id="C", agent_id="c", depends_on=["A"]),
                "D": DAGNodeData(
                    node_id="D", agent_id="d", depends_on=["B", "C"]
                ),
            },
        )

        registry = _make_deterministic_registry()
        exec1 = DAGExecutor(
            agent_registry=registry, checkpoint_store=CheckpointStore()
        )
        exec2 = DAGExecutor(
            agent_registry=registry, checkpoint_store=CheckpointStore()
        )

        result1 = _run_async(
            exec1.execute(dag, input_data={"x": 1}, execution_id="run-1")
        )
        result2 = _run_async(
            exec2.execute(dag, input_data={"x": 1}, execution_id="run-2")
        )

        assert result1.topology_levels == result2.topology_levels
        assert result1.status == result2.status

    def test_two_executions_same_provenance_hashes(self):
        dag = DAGWorkflowData(
            dag_id="prov-replay",
            name="Provenance Replay",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A"]),
            },
        )

        registry = _make_deterministic_registry()
        exec1 = DAGExecutor(
            agent_registry=registry, checkpoint_store=CheckpointStore()
        )
        exec2 = DAGExecutor(
            agent_registry=registry, checkpoint_store=CheckpointStore()
        )

        result1 = _run_async(exec1.execute(dag, execution_id="p-1"))
        result2 = _run_async(exec2.execute(dag, execution_id="p-2"))

        # Same provenance hashes for each node
        for nid in ["A", "B"]:
            assert result1.provenance_hashes[nid] == result2.provenance_hashes[nid]

    def test_linear_replay_consistency(self):
        dag = DAGWorkflowData(
            dag_id="linear-replay",
            name="Linear Replay",
            nodes={
                "step1": DAGNodeData(node_id="step1", agent_id="s1"),
                "step2": DAGNodeData(
                    node_id="step2", agent_id="s2", depends_on=["step1"]
                ),
                "step3": DAGNodeData(
                    node_id="step3", agent_id="s3", depends_on=["step2"]
                ),
            },
        )

        registry = _make_deterministic_registry()
        results = []
        for i in range(5):
            executor = DAGExecutor(
                agent_registry=registry, checkpoint_store=CheckpointStore()
            )
            r = _run_async(executor.execute(dag, execution_id=f"rep-{i}"))
            results.append(r)

        # All 5 runs should produce the same hashes
        for r in results[1:]:
            for nid in ["step1", "step2", "step3"]:
                assert (
                    r.provenance_hashes[nid] == results[0].provenance_hashes[nid]
                )


@pytest.mark.integration
class TestDifferentTracesForDifferentInputs:
    """Test that different inputs produce different execution traces."""

    def test_different_agent_outputs_different_hashes(self):
        dag = DAGWorkflowData(
            dag_id="diff-test",
            name="Different Test",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
            },
        )

        def _make_registry_v1():
            def _get(aid):
                m = MagicMock()
                m.run.return_value = {"version": 1}
                m.run_async = None
                return m
            return _get

        def _make_registry_v2():
            def _get(aid):
                m = MagicMock()
                m.run.return_value = {"version": 2}
                m.run_async = None
                return m
            return _get

        r1 = _run_async(
            DAGExecutor(
                agent_registry=_make_registry_v1(),
                checkpoint_store=CheckpointStore(),
            ).execute(dag, execution_id="d-1")
        )
        r2 = _run_async(
            DAGExecutor(
                agent_registry=_make_registry_v2(),
                checkpoint_store=CheckpointStore(),
            ).execute(dag, execution_id="d-2")
        )

        assert r1.provenance_hashes["A"] != r2.provenance_hashes["A"]


@pytest.mark.integration
class TestProvenanceChainMatchesOnReplay:
    """Test that full provenance chain matches across replays."""

    def test_chain_hash_matches(self):
        dag = DAGWorkflowData(
            dag_id="chain-test",
            name="Chain Test",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A"]),
                "C": DAGNodeData(node_id="C", agent_id="c", depends_on=["B"]),
            },
        )

        registry = _make_deterministic_registry()
        results = []
        for i in range(3):
            executor = DAGExecutor(
                agent_registry=registry, checkpoint_store=CheckpointStore()
            )
            r = _run_async(executor.execute(dag, execution_id=f"ch-{i}"))
            results.append(r)

        # Compute a combined chain hash for each run
        def _chain_hash(result):
            data = json.dumps(result.provenance_hashes, sort_keys=True)
            return hashlib.sha256(data.encode()).hexdigest()

        chain_hashes = [_chain_hash(r) for r in results]
        assert len(set(chain_hashes)) == 1  # All identical
