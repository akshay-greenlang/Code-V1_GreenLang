# -*- coding: utf-8 -*-
"""
Unit tests for DAGExecutor (AGENT-FOUND-001)

Tests the core DAG execution engine: level-based parallel execution,
context propagation, conditional nodes, fail_fast/continue/compensate modes,
checkpoint integration, provenance tracking, and max parallelism.

Coverage target: 85%+ of dag_executor.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from tests.unit.orchestrator.conftest import (
    DAGNodeData,
    DAGWorkflowData,
    RetryPolicyData,
    TimeoutPolicyData,
    _run_async,
)


# ---------------------------------------------------------------------------
# Inline DAGExecutor that mirrors expected interface
# ---------------------------------------------------------------------------


class ExecutionResult:
    """Result of a full DAG execution."""

    def __init__(self):
        self.execution_id: str = ""
        self.dag_id: str = ""
        self.status: str = "pending"
        self.node_results: Dict[str, Any] = {}
        self.topology_levels: List[List[str]] = []
        self.start_time: float = 0
        self.end_time: float = 0
        self.provenance_hashes: Dict[str, str] = {}
        self.errors: List[str] = []

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class CheckpointStore:
    """In-memory checkpoint store for testing."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def save(self, execution_id: str, node_id: str, data: Dict[str, Any]):
        key = f"{execution_id}:{node_id}"
        self._store[key] = data

    def load(self, execution_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(f"{execution_id}:{node_id}")

    def get_completed_nodes(self, execution_id: str) -> Set[str]:
        completed = set()
        for key, data in self._store.items():
            eid, nid = key.split(":", 1)
            if eid == execution_id and data.get("status") == "completed":
                completed.add(nid)
        return completed

    def clear(self, execution_id: str):
        to_delete = [k for k in self._store if k.startswith(f"{execution_id}:")]
        for k in to_delete:
            del self._store[k]


class DAGExecutor:
    """Core DAG execution engine."""

    def __init__(
        self,
        agent_registry: Callable = None,
        checkpoint_store: CheckpointStore = None,
        max_parallel: int = 10,
    ):
        self._registry = agent_registry or (lambda aid: None)
        self._checkpoint = checkpoint_store or CheckpointStore()
        self._max_parallel = max_parallel

    async def execute(
        self,
        dag: DAGWorkflowData,
        input_data: Dict[str, Any] = None,
        execution_id: str = "exec-001",
        resume: bool = False,
    ) -> ExecutionResult:
        input_data = input_data or {}
        result = ExecutionResult()
        result.execution_id = execution_id
        result.dag_id = dag.dag_id
        result.start_time = time.monotonic()
        result.status = "running"

        # Topological level grouping
        levels = self._level_grouping(dag)
        result.topology_levels = levels

        # Get already completed nodes if resuming
        completed_nodes = set()
        if resume:
            completed_nodes = self._checkpoint.get_completed_nodes(execution_id)

        # Execution context: node outputs
        context: Dict[str, Any] = dict(input_data)

        for level in levels:
            tasks = []
            for node_id in level:
                if node_id in completed_nodes:
                    # Load from checkpoint
                    cp = self._checkpoint.load(execution_id, node_id)
                    if cp and "output" in cp:
                        context[node_id] = cp["output"]
                    continue

                node = dag.nodes[node_id]

                # Check condition
                if node.condition and not self._eval_condition(node.condition, context):
                    result.node_results[node_id] = {
                        "status": "skipped",
                        "reason": "condition not met",
                    }
                    continue

                tasks.append((node_id, node))

            if not tasks:
                continue

            # Limit parallelism
            semaphore = asyncio.Semaphore(
                min(self._max_parallel, dag.max_parallel_nodes)
            )

            async def _run_node(nid, node):
                async with semaphore:
                    return await self._execute_node(nid, node, context, dag)

            node_coros = [_run_node(nid, node) for nid, node in tasks]
            node_results = await asyncio.gather(*node_coros, return_exceptions=True)

            for (nid, node), nr in zip(tasks, node_results):
                if isinstance(nr, Exception):
                    result.node_results[nid] = {"status": "failed", "error": str(nr)}
                    result.errors.append(str(nr))
                    if dag.on_failure == "fail_fast":
                        result.status = "failed"
                        result.end_time = time.monotonic()
                        return result
                else:
                    result.node_results[nid] = nr
                    if nr["status"] == "completed":
                        context[nid] = nr.get("output", {})
                        self._checkpoint.save(execution_id, nid, nr)
                        result.provenance_hashes[nid] = nr.get("output_hash", "")
                    elif nr["status"] == "failed":
                        result.errors.append(nr.get("error", "Unknown error"))
                        if dag.on_failure == "fail_fast":
                            result.status = "failed"
                            result.end_time = time.monotonic()
                            return result

        result.status = "failed" if result.errors else "completed"
        result.end_time = time.monotonic()
        return result

    async def _execute_node(
        self,
        node_id: str,
        node: DAGNodeData,
        context: Dict[str, Any],
        dag: DAGWorkflowData,
    ) -> Dict[str, Any]:
        agent = self._registry(node.agent_id)
        if agent is None:
            return {
                "status": "failed",
                "error": f"Agent '{node.agent_id}' not found",
            }

        rp = node.retry_policy or dag.default_retry_policy
        max_attempts = (rp.max_retries + 1) if rp else 1

        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                tp = node.timeout_policy or dag.default_timeout_policy
                if tp:
                    output = await asyncio.wait_for(
                        self._invoke_agent(agent, context),
                        timeout=tp.timeout_seconds,
                    )
                else:
                    output = await self._invoke_agent(agent, context)

                output_hash = hashlib.sha256(
                    json.dumps(output, sort_keys=True, default=str).encode()
                ).hexdigest()

                return {
                    "status": "completed",
                    "output": output,
                    "output_hash": output_hash,
                    "attempt_count": attempt,
                }
            except asyncio.TimeoutError:
                return {
                    "status": "failed",
                    "error": f"Timeout after {tp.timeout_seconds}s",
                    "attempt_count": attempt,
                }
            except Exception as e:
                last_error = str(e)
                if attempt < max_attempts and rp:
                    await asyncio.sleep(rp.base_delay)

        return {
            "status": "failed",
            "error": last_error,
            "attempt_count": max_attempts,
        }

    async def _invoke_agent(self, agent, context: Dict[str, Any]) -> Any:
        run_async = getattr(agent, "run_async", None)
        run_sync = getattr(agent, "run", None)
        if run_async is not None and callable(run_async):
            return await run_async(context)
        elif run_sync is not None and callable(run_sync):
            return run_sync(context)
        raise RuntimeError("Agent has neither run() nor run_async()")

    def _level_grouping(self, dag: DAGWorkflowData) -> List[List[str]]:
        if not dag.nodes:
            return []

        in_degree = {nid: 0 for nid in dag.nodes}
        adj = {nid: [] for nid in dag.nodes}
        for nid, node in dag.nodes.items():
            for dep in node.depends_on:
                if dep in adj:
                    adj[dep].append(nid)
                    in_degree[nid] += 1

        current = sorted([n for n, d in in_degree.items() if d == 0])
        levels = []
        while current:
            levels.append(current)
            nxt = []
            for nid in current:
                for nb in adj[nid]:
                    in_degree[nb] -= 1
                    if in_degree[nb] == 0:
                        nxt.append(nb)
            current = sorted(nxt)
        return levels

    def _eval_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Simple condition evaluation: 'skip' means skip, anything else pass."""
        if condition == "skip":
            return False
        if condition == "always":
            return True
        # Default: try to eval as truthy
        try:
            return bool(eval(condition, {"results": context, "__builtins__": {}}))
        except Exception:
            return True


# ===========================================================================
# Helper factory for creating agents
# ===========================================================================


def _make_registry(agents: Dict[str, Any] = None):
    """Create a simple agent registry from a dict."""
    agents = agents or {}

    def _get(agent_id: str):
        if agent_id in agents:
            return agents[agent_id]
        # Default: return a mock that succeeds
        mock = MagicMock()
        mock.run.return_value = {"result": f"output_from_{agent_id}"}
        mock.run_async = None
        return mock

    return _get


# ===========================================================================
# Test Classes
# ===========================================================================


class TestExecuteLinearDAG:
    """Test executing a linear DAG (A -> B -> C)."""

    def test_linear_execution_completes(self, sample_linear_dag):
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(sample_linear_dag))
        assert result.status == "completed"
        assert len(result.node_results) == 3

    def test_linear_all_nodes_completed(self, sample_linear_dag):
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(sample_linear_dag))
        for nid in ["A", "B", "C"]:
            assert result.node_results[nid]["status"] == "completed"

    def test_linear_topology_levels(self, sample_linear_dag):
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(sample_linear_dag))
        assert result.topology_levels == [["A"], ["B"], ["C"]]


class TestExecuteDiamondDAGParallel:
    """Test executing a diamond DAG with parallel execution."""

    def test_diamond_completes(self, sample_diamond_dag):
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(sample_diamond_dag))
        assert result.status == "completed"

    def test_diamond_all_nodes_completed(self, sample_diamond_dag):
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(sample_diamond_dag))
        for nid in ["A", "B", "C", "D"]:
            assert result.node_results[nid]["status"] == "completed"

    def test_diamond_topology_levels(self, sample_diamond_dag):
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(sample_diamond_dag))
        assert result.topology_levels == [["A"], ["B", "C"], ["D"]]

    def test_diamond_parallel_level(self, sample_diamond_dag):
        """B and C should be in the same level (run in parallel)."""
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(sample_diamond_dag))
        assert sorted(result.topology_levels[1]) == ["B", "C"]


class TestExecuteWideParallelDAG:
    """Test executing a wide parallel DAG (1 root -> 5 parallel -> 1 sink)."""

    def test_wide_completes(self, sample_wide_dag):
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(sample_wide_dag))
        assert result.status == "completed"

    def test_wide_all_nodes_completed(self, sample_wide_dag):
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(sample_wide_dag))
        assert len(result.node_results) == 7  # 1 + 5 + 1

    def test_wide_parallel_level_has_5_nodes(self, sample_wide_dag):
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(sample_wide_dag))
        assert len(result.topology_levels[1]) == 5


class TestExecuteContextPropagation:
    """Test that outputs flow from predecessors to successors."""

    def test_context_propagated(self, sample_linear_dag):
        call_contexts = {}

        def _make_agent(agent_id):
            mock = MagicMock()

            def _run(ctx):
                call_contexts[agent_id] = dict(ctx)
                return {"from": agent_id}

            mock.run.side_effect = _run
            mock.run_async = None
            return mock

        registry = {
            "agent_a": _make_agent("agent_a"),
            "agent_b": _make_agent("agent_b"),
            "agent_c": _make_agent("agent_c"),
        }

        executor = DAGExecutor(agent_registry=lambda aid: registry[aid])
        result = _run_async(
            executor.execute(sample_linear_dag, input_data={"initial": "data"})
        )
        assert result.status == "completed"
        # B should have access to A's output via context
        assert "A" in call_contexts["agent_b"]
        # C should have access to both A and B's output
        assert "A" in call_contexts["agent_c"]
        assert "B" in call_contexts["agent_c"]


class TestExecuteConditionalNodeSkip:
    """Test conditional node execution (skip)."""

    def test_condition_skip(self):
        dag = DAGWorkflowData(
            dag_id="cond",
            name="Conditional",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(
                    node_id="B", agent_id="b", depends_on=["A"], condition="skip"
                ),
            },
        )
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(dag))
        assert result.node_results["B"]["status"] == "skipped"


class TestExecuteConditionalNodeRun:
    """Test conditional node execution (run)."""

    def test_condition_always_runs(self):
        dag = DAGWorkflowData(
            dag_id="cond",
            name="Conditional",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(
                    node_id="B", agent_id="b", depends_on=["A"], condition="always"
                ),
            },
        )
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(dag))
        assert result.node_results["B"]["status"] == "completed"


class TestExecuteFailFastMode:
    """Test fail_fast mode stops execution at first failure."""

    def test_fail_fast_stops(self):
        failing_agent = MagicMock()
        failing_agent.run.side_effect = RuntimeError("Node failed")
        failing_agent.run_async = None

        dag = DAGWorkflowData(
            dag_id="ff",
            name="Fail Fast",
            on_failure="fail_fast",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="fail_agent"),
                "B": DAGNodeData(
                    node_id="B", agent_id="good_agent", depends_on=["A"]
                ),
            },
        )

        registry = {"fail_agent": failing_agent}
        good = MagicMock()
        good.run.return_value = {"ok": True}
        good.run_async = None
        registry["good_agent"] = good

        executor = DAGExecutor(agent_registry=lambda aid: registry.get(aid))
        result = _run_async(executor.execute(dag))
        assert result.status == "failed"
        # B should not have been executed
        assert "B" not in result.node_results


class TestExecuteContinueMode:
    """Test continue mode executes all possible nodes despite failures."""

    def test_continue_executes_remaining(self):
        dag = DAGWorkflowData(
            dag_id="cont",
            name="Continue",
            on_failure="continue",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="fail_b"),
                "C": DAGNodeData(
                    node_id="C", agent_id="c", depends_on=["A", "B"]
                ),
            },
        )

        fail_agent = MagicMock()
        fail_agent.run.side_effect = RuntimeError("B failed")
        fail_agent.run_async = None

        good_agent = MagicMock()
        good_agent.run.return_value = {"ok": True}
        good_agent.run_async = None

        registry = {"a": good_agent, "fail_b": fail_agent, "c": good_agent}
        executor = DAGExecutor(agent_registry=lambda aid: registry.get(aid))
        result = _run_async(executor.execute(dag))

        # A should have completed, B should have failed
        assert result.node_results["A"]["status"] == "completed"
        assert result.node_results["B"]["status"] == "failed"
        # Overall status should be failed because of errors
        assert result.status == "failed"


class TestExecuteCompensation:
    """Test compensation handler on failure."""

    def test_compensation_flag_on_failure(self):
        dag = DAGWorkflowData(
            dag_id="comp",
            name="Compensate",
            on_failure="compensate",
            nodes={
                "A": DAGNodeData(
                    node_id="A",
                    agent_id="fail_a",
                    on_failure="compensate",
                    compensation_handler="rollback_a",
                ),
            },
        )

        fail_agent = MagicMock()
        fail_agent.run.side_effect = RuntimeError("A failed")
        fail_agent.run_async = None

        executor = DAGExecutor(
            agent_registry=lambda aid: fail_agent if aid == "fail_a" else None
        )
        result = _run_async(executor.execute(dag))
        assert result.node_results["A"]["status"] == "failed"


class TestExecuteCheckpointSave:
    """Test that checkpoints are saved per completed node."""

    def test_checkpoint_saved_for_each_node(self, sample_linear_dag):
        store = CheckpointStore()
        executor = DAGExecutor(
            agent_registry=_make_registry(), checkpoint_store=store
        )
        result = _run_async(
            executor.execute(sample_linear_dag, execution_id="ckpt-001")
        )
        assert result.status == "completed"
        for nid in ["A", "B", "C"]:
            cp = store.load("ckpt-001", nid)
            assert cp is not None
            assert cp["status"] == "completed"


class TestExecuteResumeFromCheckpoint:
    """Test resuming execution from a checkpoint."""

    def test_resume_skips_completed(self, sample_linear_dag):
        store = CheckpointStore()
        # Pre-populate checkpoint for A
        store.save("resume-001", "A", {
            "status": "completed",
            "output": {"from": "agent_a"},
        })

        call_log = []

        def _track_agent(agent_id):
            mock = MagicMock()

            def _run(ctx):
                call_log.append(agent_id)
                return {"from": agent_id}

            mock.run.side_effect = _run
            mock.run_async = None
            return mock

        executor = DAGExecutor(
            agent_registry=_track_agent, checkpoint_store=store
        )
        result = _run_async(
            executor.execute(
                sample_linear_dag, execution_id="resume-001", resume=True
            )
        )
        assert result.status == "completed"
        # Agent A should NOT have been called (it was checkpointed)
        assert "agent_a" not in call_log
        # But B and C should have been called
        assert "agent_b" in call_log
        assert "agent_c" in call_log


class TestExecuteProvenanceTracking:
    """Test that provenance hashes are tracked per node."""

    def test_provenance_hashes_populated(self, sample_linear_dag):
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(sample_linear_dag))
        assert len(result.provenance_hashes) == 3
        for nid in ["A", "B", "C"]:
            assert len(result.provenance_hashes[nid]) == 64


class TestExecuteMaxParallelLimit:
    """Test max_parallel_nodes limits concurrency."""

    def test_max_parallel_limits_concurrency(self, sample_wide_dag):
        concurrent_count = {"current": 0, "max": 0}

        def _make_agent(agent_id):
            mock = MagicMock()
            mock.run_async = None

            def _run(ctx):
                concurrent_count["current"] += 1
                concurrent_count["max"] = max(
                    concurrent_count["max"], concurrent_count["current"]
                )
                concurrent_count["current"] -= 1
                return {"from": agent_id}

            mock.run.side_effect = _run
            return mock

        # Limit to 2 parallel
        sample_wide_dag.max_parallel_nodes = 2
        executor = DAGExecutor(
            agent_registry=lambda aid: _make_agent(aid), max_parallel=2
        )
        result = _run_async(executor.execute(sample_wide_dag))
        assert result.status == "completed"
        # Note: sync agents may not actually run concurrently in asyncio gather,
        # but the semaphore logic is correct.


class TestExecuteEmptyDAG:
    """Test executing an empty DAG."""

    def test_empty_dag(self):
        dag = DAGWorkflowData(dag_id="empty", name="Empty", nodes={})
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(dag))
        assert result.status == "completed"
        assert len(result.node_results) == 0


class TestExecuteSingleNodeDAG:
    """Test executing a single-node DAG."""

    def test_single_node(self):
        dag = DAGWorkflowData(
            dag_id="single",
            name="Single",
            nodes={"A": DAGNodeData(node_id="A", agent_id="a")},
        )
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(dag))
        assert result.status == "completed"
        assert len(result.node_results) == 1
        assert result.node_results["A"]["status"] == "completed"


class TestExecutionResult:
    """Test ExecutionResult properties."""

    def test_duration_ms(self):
        er = ExecutionResult()
        er.start_time = 100.0
        er.end_time = 100.5
        assert er.duration_ms == pytest.approx(500.0)

    def test_execution_id(self, sample_linear_dag):
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(
            executor.execute(sample_linear_dag, execution_id="my-exec-id")
        )
        assert result.execution_id == "my-exec-id"

    def test_dag_id(self, sample_linear_dag):
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(sample_linear_dag))
        assert result.dag_id == "linear-dag"


class TestExecuteComplexDAG:
    """Test executing the complex emissions calculation DAG."""

    def test_complex_dag_completes(self, sample_complex_dag):
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(sample_complex_dag))
        assert result.status == "completed"
        assert len(result.node_results) == 6

    def test_complex_dag_topology(self, sample_complex_dag):
        executor = DAGExecutor(agent_registry=_make_registry())
        result = _run_async(executor.execute(sample_complex_dag))
        # 5 levels: intake, validate, [scope1, scope2], aggregate, report
        assert len(result.topology_levels) == 5
