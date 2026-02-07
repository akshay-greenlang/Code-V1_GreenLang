# -*- coding: utf-8 -*-
"""
End-to-End Integration Tests for DAG Orchestrator (AGENT-FOUND-001)

Tests full DAG execution flows: linear, diamond, complex, result
propagation, and failure handling across the complete pipeline.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock

import pytest

from tests.integration.orchestrator.conftest import _run_async


# ---------------------------------------------------------------------------
# Import the inline executor from unit tests for integration testing
# (In production, this would import from greenlang.orchestrator.dag_executor)
# ---------------------------------------------------------------------------

from tests.unit.orchestrator.test_dag_executor import (
    CheckpointStore,
    DAGExecutor,
    ExecutionResult,
)
from tests.unit.orchestrator.conftest import (
    DAGNodeData,
    DAGWorkflowData,
    RetryPolicyData,
    TimeoutPolicyData,
)


def _make_registry(agent_map: Dict[str, Any] = None):
    """Create agent registry from map, with default fallback."""
    agent_map = agent_map or {}

    def _get(agent_id: str):
        if agent_id in agent_map:
            return agent_map[agent_id]
        mock = MagicMock()
        mock.run.return_value = {"result": f"output_from_{agent_id}"}
        mock.run_async = None
        return mock

    return _get


# ===========================================================================
# Test Classes
# ===========================================================================


@pytest.mark.integration
class TestFullLinearDAGExecution:
    """Full end-to-end test of linear DAG execution."""

    def test_linear_dag_completes_all_nodes(self):
        dag = DAGWorkflowData(
            dag_id="e2e-linear",
            name="E2E Linear",
            nodes={
                "intake": DAGNodeData(node_id="intake", agent_id="intake_agent"),
                "process": DAGNodeData(
                    node_id="process", agent_id="process_agent", depends_on=["intake"]
                ),
                "output": DAGNodeData(
                    node_id="output", agent_id="output_agent", depends_on=["process"]
                ),
            },
        )
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(dag, input_data={"facility": "FAC-001"}))
        assert result.status == "completed"
        assert len(result.node_results) == 3
        for nid in ["intake", "process", "output"]:
            assert result.node_results[nid]["status"] == "completed"

    def test_linear_dag_execution_time_reasonable(self):
        dag = DAGWorkflowData(
            dag_id="e2e-perf",
            name="E2E Perf",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A"]),
                "C": DAGNodeData(node_id="C", agent_id="c", depends_on=["B"]),
            },
        )
        executor = DAGExecutor(agent_registry=_make_registry())
        start = time.monotonic()
        result = _run_async(executor.execute(dag))
        elapsed = time.monotonic() - start
        assert result.status == "completed"
        assert elapsed < 5.0  # Should complete in well under 5 seconds

    def test_linear_dag_has_provenance(self):
        dag = DAGWorkflowData(
            dag_id="e2e-prov",
            name="E2E Provenance",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A"]),
            },
        )
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(dag))
        assert len(result.provenance_hashes) == 2
        for h in result.provenance_hashes.values():
            assert len(h) == 64


@pytest.mark.integration
class TestFullDiamondDAGExecution:
    """Full end-to-end test of diamond DAG execution."""

    def test_diamond_dag_completes(self):
        dag = DAGWorkflowData(
            dag_id="e2e-diamond",
            name="E2E Diamond",
            nodes={
                "start": DAGNodeData(node_id="start", agent_id="start_agent"),
                "branch_a": DAGNodeData(
                    node_id="branch_a", agent_id="a_agent", depends_on=["start"]
                ),
                "branch_b": DAGNodeData(
                    node_id="branch_b", agent_id="b_agent", depends_on=["start"]
                ),
                "merge": DAGNodeData(
                    node_id="merge",
                    agent_id="merge_agent",
                    depends_on=["branch_a", "branch_b"],
                ),
            },
        )
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(dag))
        assert result.status == "completed"
        assert len(result.node_results) == 4

    def test_diamond_parallel_level(self):
        dag = DAGWorkflowData(
            dag_id="e2e-diamond-lvl",
            name="E2E Diamond Levels",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A"]),
                "C": DAGNodeData(node_id="C", agent_id="c", depends_on=["A"]),
                "D": DAGNodeData(node_id="D", agent_id="d", depends_on=["B", "C"]),
            },
        )
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(dag))
        assert result.topology_levels[1] == ["B", "C"]


@pytest.mark.integration
class TestFullComplexDAGExecution:
    """Full end-to-end test of complex emissions DAG."""

    def test_emissions_dag_completes(self):
        dag = DAGWorkflowData(
            dag_id="emissions-calc",
            name="Emissions Calculation",
            nodes={
                "intake": DAGNodeData(node_id="intake", agent_id="intake_agent"),
                "validate": DAGNodeData(
                    node_id="validate", agent_id="validate_agent", depends_on=["intake"]
                ),
                "scope1": DAGNodeData(
                    node_id="scope1", agent_id="scope1_agent", depends_on=["validate"]
                ),
                "scope2": DAGNodeData(
                    node_id="scope2", agent_id="scope2_agent", depends_on=["validate"]
                ),
                "aggregate": DAGNodeData(
                    node_id="aggregate",
                    agent_id="aggregate_agent",
                    depends_on=["scope1", "scope2"],
                ),
                "report": DAGNodeData(
                    node_id="report", agent_id="report_agent", depends_on=["aggregate"]
                ),
            },
            default_retry_policy=RetryPolicyData(max_retries=1, base_delay=0.001),
        )
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(dag))
        assert result.status == "completed"
        assert len(result.node_results) == 6

    def test_emissions_dag_topology(self):
        dag = DAGWorkflowData(
            dag_id="emissions-topo",
            name="Emissions Topology",
            nodes={
                "intake": DAGNodeData(node_id="intake", agent_id="a"),
                "validate": DAGNodeData(
                    node_id="validate", agent_id="b", depends_on=["intake"]
                ),
                "scope1": DAGNodeData(
                    node_id="scope1", agent_id="c", depends_on=["validate"]
                ),
                "scope2": DAGNodeData(
                    node_id="scope2", agent_id="d", depends_on=["validate"]
                ),
                "aggregate": DAGNodeData(
                    node_id="aggregate",
                    agent_id="e",
                    depends_on=["scope1", "scope2"],
                ),
                "report": DAGNodeData(
                    node_id="report", agent_id="f", depends_on=["aggregate"]
                ),
            },
        )
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(dag))
        assert len(result.topology_levels) == 5


@pytest.mark.integration
class TestExecutionResultPropagation:
    """Test that results propagate correctly between nodes."""

    def test_predecessor_output_available_to_successor(self):
        captured_contexts = {}

        def _make_capturing_agent(agent_id):
            mock = MagicMock()
            mock.run_async = None

            def _run(ctx):
                captured_contexts[agent_id] = dict(ctx)
                return {"from": agent_id, "value": len(ctx)}

            mock.run.side_effect = _run
            return mock

        registry = {
            "agent_a": _make_capturing_agent("agent_a"),
            "agent_b": _make_capturing_agent("agent_b"),
        }

        dag = DAGWorkflowData(
            dag_id="prop-test",
            name="Propagation Test",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="agent_a"),
                "B": DAGNodeData(
                    node_id="B", agent_id="agent_b", depends_on=["A"]
                ),
            },
        )

        executor = DAGExecutor(agent_registry=lambda aid: registry[aid])
        result = _run_async(
            executor.execute(dag, input_data={"initial": "data"})
        )
        assert result.status == "completed"
        # B's context should include A's output
        assert "A" in captured_contexts["agent_b"]

    def test_diamond_both_branches_available_at_merge(self):
        captured = {}

        def _make_agent(agent_id):
            mock = MagicMock()
            mock.run_async = None

            def _run(ctx):
                captured[agent_id] = set(ctx.keys())
                return {"from": agent_id}

            mock.run.side_effect = _run
            return mock

        agents = {aid: _make_agent(aid) for aid in ["a", "b", "c", "d"]}
        dag = DAGWorkflowData(
            dag_id="merge-test",
            name="Merge Test",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A"]),
                "C": DAGNodeData(node_id="C", agent_id="c", depends_on=["A"]),
                "D": DAGNodeData(
                    node_id="D", agent_id="d", depends_on=["B", "C"]
                ),
            },
        )

        executor = DAGExecutor(agent_registry=lambda aid: agents[aid])
        result = _run_async(executor.execute(dag))
        assert result.status == "completed"
        # D should have access to A, B, and C
        assert "A" in captured["d"]
        assert "B" in captured["d"]
        assert "C" in captured["d"]


@pytest.mark.integration
class TestExecutionWithFailingNode:
    """Test execution behavior when a node fails."""

    def test_fail_fast_stops_at_failure(self):
        fail_mock = MagicMock()
        fail_mock.run.side_effect = RuntimeError("Agent crashed")
        fail_mock.run_async = None

        good_mock = MagicMock()
        good_mock.run.return_value = {"ok": True}
        good_mock.run_async = None

        dag = DAGWorkflowData(
            dag_id="fail-test",
            name="Fail Test",
            on_failure="fail_fast",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="fail_agent"),
                "B": DAGNodeData(
                    node_id="B", agent_id="good_agent", depends_on=["A"]
                ),
            },
        )

        executor = DAGExecutor(
            agent_registry=lambda aid: fail_mock
            if aid == "fail_agent"
            else good_mock
        )
        result = _run_async(executor.execute(dag))
        assert result.status == "failed"
        assert len(result.errors) > 0

    def test_retry_recovers_from_transient_failure(self):
        call_count = {"n": 0}

        def _flaky_run(ctx):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                raise RuntimeError("Transient failure")
            return {"recovered": True}

        flaky = MagicMock()
        flaky.run.side_effect = _flaky_run
        flaky.run_async = None

        dag = DAGWorkflowData(
            dag_id="retry-test",
            name="Retry Test",
            nodes={
                "A": DAGNodeData(
                    node_id="A",
                    agent_id="flaky",
                    retry_policy=RetryPolicyData(
                        max_retries=3, strategy="constant", base_delay=0.001
                    ),
                ),
            },
        )

        executor = DAGExecutor(agent_registry=lambda aid: flaky)
        result = _run_async(executor.execute(dag))
        assert result.status == "completed"
        assert result.node_results["A"]["attempt_count"] == 2
