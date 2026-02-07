# -*- coding: utf-8 -*-
"""
Checkpoint Resume Integration Tests (AGENT-FOUND-001)

Tests checkpoint/resume flow: execute, fail at a node, resume from
checkpoint, verify completed nodes are skipped, and failed node re-executes.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.integration.orchestrator.conftest import _run_async
from tests.unit.orchestrator.test_dag_executor import CheckpointStore, DAGExecutor
from tests.unit.orchestrator.conftest import (
    DAGNodeData,
    DAGWorkflowData,
    RetryPolicyData,
)


def _make_registry(agent_map=None):
    agent_map = agent_map or {}

    def _get(agent_id):
        if agent_id in agent_map:
            return agent_map[agent_id]
        mock = MagicMock()
        mock.run.return_value = {"result": f"from_{agent_id}"}
        mock.run_async = None
        return mock

    return _get


@pytest.mark.integration
class TestExecuteFailResume:
    """Test execute -> fail -> resume flow."""

    def test_fail_then_resume_completes(self):
        call_count = {"n": 0}

        def _fail_first(ctx):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                raise RuntimeError("First attempt fails")
            return {"recovered": True}

        fail_agent = MagicMock()
        fail_agent.run.side_effect = _fail_first
        fail_agent.run_async = None

        good = MagicMock()
        good.run.return_value = {"ok": True}
        good.run_async = None

        dag = DAGWorkflowData(
            dag_id="resume-test",
            name="Resume Test",
            on_failure="fail_fast",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="good"),
                "B": DAGNodeData(
                    node_id="B", agent_id="fail_then_succeed", depends_on=["A"]
                ),
                "C": DAGNodeData(
                    node_id="C", agent_id="good", depends_on=["B"]
                ),
            },
        )

        store = CheckpointStore()
        executor = DAGExecutor(
            agent_registry=lambda aid: fail_agent
            if aid == "fail_then_succeed"
            else good,
            checkpoint_store=store,
        )

        # First execution: A completes, B fails, C never runs
        result1 = _run_async(
            executor.execute(dag, execution_id="r-001")
        )
        assert result1.status == "failed"
        assert store.get_completed_nodes("r-001") == {"A"}

        # Resume: A skipped (checkpointed), B retries and succeeds, C runs
        result2 = _run_async(
            executor.execute(dag, execution_id="r-001", resume=True)
        )
        assert result2.status == "completed"

    def test_resume_with_all_nodes_completed(self):
        """If all nodes are already checkpointed, resume returns immediately."""
        store = CheckpointStore()
        good = MagicMock()
        good.run.return_value = {"ok": True}
        good.run_async = None

        dag = DAGWorkflowData(
            dag_id="all-done",
            name="All Done",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="good"),
                "B": DAGNodeData(
                    node_id="B", agent_id="good", depends_on=["A"]
                ),
            },
        )

        # Pre-populate all checkpoints
        store.save("done-001", "A", {"status": "completed", "output": {"ok": True}})
        store.save("done-001", "B", {"status": "completed", "output": {"ok": True}})

        executor = DAGExecutor(
            agent_registry=lambda aid: good, checkpoint_store=store
        )
        result = _run_async(
            executor.execute(dag, execution_id="done-001", resume=True)
        )
        assert result.status == "completed"
        # No new node results should be generated (all from checkpoint)
        assert len(result.node_results) == 0


@pytest.mark.integration
class TestResumeSkipsCompletedNodes:
    """Test that resume skips already-completed nodes."""

    def test_completed_nodes_not_re_executed(self):
        call_log = []

        def _tracking_agent(agent_id):
            mock = MagicMock()
            mock.run_async = None

            def _run(ctx):
                call_log.append(agent_id)
                return {"from": agent_id}

            mock.run.side_effect = _run
            return mock

        dag = DAGWorkflowData(
            dag_id="skip-test",
            name="Skip Test",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="agent_a"),
                "B": DAGNodeData(
                    node_id="B", agent_id="agent_b", depends_on=["A"]
                ),
                "C": DAGNodeData(
                    node_id="C", agent_id="agent_c", depends_on=["B"]
                ),
            },
        )

        store = CheckpointStore()
        store.save("skip-001", "A", {
            "status": "completed",
            "output": {"from": "agent_a"},
        })

        executor = DAGExecutor(
            agent_registry=_tracking_agent, checkpoint_store=store
        )
        result = _run_async(
            executor.execute(dag, execution_id="skip-001", resume=True)
        )
        assert result.status == "completed"
        assert "agent_a" not in call_log  # A was skipped
        assert "agent_b" in call_log
        assert "agent_c" in call_log


@pytest.mark.integration
class TestResumeReExecutesFailedNode:
    """Test that resume re-executes the previously failed node."""

    def test_failed_node_runs_again(self):
        execution_count = {"B": 0}

        def _agent(agent_id):
            mock = MagicMock()
            mock.run_async = None

            def _run(ctx):
                if agent_id == "agent_b":
                    execution_count["B"] += 1
                return {"from": agent_id}

            mock.run.side_effect = _run
            return mock

        dag = DAGWorkflowData(
            dag_id="rerun-test",
            name="Rerun Test",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="agent_a"),
                "B": DAGNodeData(
                    node_id="B", agent_id="agent_b", depends_on=["A"]
                ),
            },
        )

        store = CheckpointStore()
        # A completed, B not in checkpoint (it failed)
        store.save("rerun-001", "A", {
            "status": "completed",
            "output": {"from": "agent_a"},
        })

        executor = DAGExecutor(
            agent_registry=_agent, checkpoint_store=store
        )
        result = _run_async(
            executor.execute(dag, execution_id="rerun-001", resume=True)
        )
        assert result.status == "completed"
        assert execution_count["B"] == 1


@pytest.mark.integration
class TestCheckpointIntegrityOnResume:
    """Test checkpoint integrity verification on resume."""

    def test_valid_checkpoint_loads(self):
        store = CheckpointStore()
        store.save("int-001", "A", {
            "status": "completed",
            "output": {"val": 42},
        })

        good = MagicMock()
        good.run.return_value = {"ok": True}
        good.run_async = None

        dag = DAGWorkflowData(
            dag_id="int-test",
            name="Integrity Test",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="good"),
                "B": DAGNodeData(
                    node_id="B", agent_id="good", depends_on=["A"]
                ),
            },
        )

        executor = DAGExecutor(
            agent_registry=lambda aid: good, checkpoint_store=store
        )
        result = _run_async(
            executor.execute(dag, execution_id="int-001", resume=True)
        )
        assert result.status == "completed"

    def test_multiple_checkpoints_all_loaded(self):
        store = CheckpointStore()
        for nid in ["A", "B", "C"]:
            store.save("multi-001", nid, {
                "status": "completed",
                "output": {"from": nid},
            })

        assert store.get_completed_nodes("multi-001") == {"A", "B", "C"}
