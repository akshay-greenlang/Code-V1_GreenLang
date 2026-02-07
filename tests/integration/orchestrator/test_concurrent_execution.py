# -*- coding: utf-8 -*-
"""
Concurrent Execution Integration Tests (AGENT-FOUND-001)

Tests multiple concurrent DAG executions for isolation and correctness.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from tests.unit.orchestrator.test_dag_executor import (
    CheckpointStore,
    DAGExecutor,
)
from tests.unit.orchestrator.conftest import DAGNodeData, DAGWorkflowData


def _make_registry():
    def _get(agent_id):
        mock = MagicMock()
        mock.run.return_value = {"result": f"from_{agent_id}"}
        mock.run_async = None
        return mock
    return _get


@pytest.mark.integration
class TestConcurrentDAGExecutions:
    """Test multiple DAG executions running concurrently."""

    def test_10_concurrent_dag_executions(self):
        """Execute 10 DAGs concurrently, all should complete."""
        dag = DAGWorkflowData(
            dag_id="concurrent-dag",
            name="Concurrent DAG",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A"]),
                "C": DAGNodeData(node_id="C", agent_id="c", depends_on=["B"]),
            },
        )

        async def _run_all():
            executors = [
                DAGExecutor(
                    agent_registry=_make_registry(),
                    checkpoint_store=CheckpointStore(),
                )
                for _ in range(10)
            ]
            tasks = [
                ex.execute(dag, execution_id=f"conc-{i}")
                for i, ex in enumerate(executors)
            ]
            return await asyncio.gather(*tasks)

        if sys.platform == "win32":
            loop = asyncio.new_event_loop()
            try:
                results = loop.run_until_complete(_run_all())
            finally:
                loop.close()
        else:
            results = asyncio.get_event_loop().run_until_complete(_run_all())

        assert len(results) == 10
        for result in results:
            assert result.status == "completed"
            assert len(result.node_results) == 3

    def test_concurrent_execution_isolation(self):
        """Each execution should have independent state."""
        captured_contexts: Dict[str, Dict[str, Any]] = {}

        def _make_capturing_registry(exec_id):
            def _get(agent_id):
                mock = MagicMock()
                mock.run_async = None

                def _run(ctx):
                    key = f"{exec_id}:{agent_id}"
                    captured_contexts[key] = dict(ctx)
                    return {"from": agent_id, "exec": exec_id}

                mock.run.side_effect = _run
                return mock
            return _get

        dag = DAGWorkflowData(
            dag_id="isolation-test",
            name="Isolation Test",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(
                    node_id="B", agent_id="b", depends_on=["A"]
                ),
            },
        )

        async def _run_isolated():
            tasks = []
            for i in range(5):
                ex = DAGExecutor(
                    agent_registry=_make_capturing_registry(f"exec-{i}"),
                    checkpoint_store=CheckpointStore(),
                )
                tasks.append(
                    ex.execute(
                        dag,
                        input_data={"exec_id": f"exec-{i}"},
                        execution_id=f"exec-{i}",
                    )
                )
            return await asyncio.gather(*tasks)

        if sys.platform == "win32":
            loop = asyncio.new_event_loop()
            try:
                results = loop.run_until_complete(_run_isolated())
            finally:
                loop.close()
        else:
            results = asyncio.get_event_loop().run_until_complete(_run_isolated())

        assert len(results) == 5
        for result in results:
            assert result.status == "completed"

        # Verify isolation: each exec-N:b should see its own exec_id
        for i in range(5):
            key = f"exec-{i}:b"
            assert key in captured_contexts
            assert captured_contexts[key].get("exec_id") == f"exec-{i}"
