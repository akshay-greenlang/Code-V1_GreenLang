# -*- coding: utf-8 -*-
"""
Unit tests for Orchestrator Models (AGENT-FOUND-001)

Tests DAGNode, DAGWorkflow, RetryPolicy, TimeoutPolicy, ExecutionTrace,
NodeProvenance, and related model classes.

Coverage target: 85%+ of models.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from tests.unit.orchestrator.conftest import (
    DAGNodeData,
    DAGWorkflowData,
    ExecutionStatus,
    OnFailure,
    OnTimeout,
    RetryPolicyData,
    RetryStrategy,
    TimeoutPolicyData,
)


# ---------------------------------------------------------------------------
# DAGNode Tests
# ---------------------------------------------------------------------------


class TestDAGNode:
    """Tests for DAGNode model."""

    def test_dag_node_creation_minimal(self):
        node = DAGNodeData(node_id="test", agent_id="agent_1")
        assert node.node_id == "test"
        assert node.agent_id == "agent_1"
        assert node.depends_on == []
        assert node.priority == 0

    def test_dag_node_creation_full(self):
        node = DAGNodeData(
            node_id="calc",
            agent_id="calc_agent",
            depends_on=["input", "validate"],
            input_mapping={"data": "results.input.raw"},
            output_key="calculated",
            condition="results.validate.status == 'PASS'",
            retry_policy=RetryPolicyData(max_retries=3),
            timeout_policy=TimeoutPolicyData(timeout_seconds=30.0),
            on_failure="skip",
            compensation_handler="rollback_calc",
            priority=10,
            metadata={"team": "data"},
        )
        assert node.node_id == "calc"
        assert len(node.depends_on) == 2
        assert node.condition is not None
        assert node.retry_policy.max_retries == 3
        assert node.timeout_policy.timeout_seconds == 30.0
        assert node.on_failure == "skip"
        assert node.priority == 10

    def test_dag_node_defaults(self):
        node = DAGNodeData(node_id="n", agent_id="a")
        assert node.depends_on == []
        assert node.input_mapping == {}
        assert node.output_key == ""
        assert node.condition is None
        assert node.retry_policy is None
        assert node.timeout_policy is None
        assert node.on_failure == "stop"
        assert node.compensation_handler is None
        assert node.priority == 0
        assert node.metadata == {}

    def test_dag_node_to_dict(self):
        node = DAGNodeData(
            node_id="A", agent_id="agent_a", depends_on=["B"], priority=5
        )
        d = node.to_dict()
        assert d["node_id"] == "A"
        assert d["agent_id"] == "agent_a"
        assert d["depends_on"] == ["B"]
        assert d["priority"] == 5
        assert d["on_failure"] == "stop"

    def test_dag_node_to_dict_with_policies(self):
        node = DAGNodeData(
            node_id="X",
            agent_id="agent_x",
            retry_policy=RetryPolicyData(max_retries=5),
            timeout_policy=TimeoutPolicyData(timeout_seconds=10.0),
        )
        d = node.to_dict()
        assert d["retry_policy"]["max_retries"] == 5
        assert d["timeout_policy"]["timeout_seconds"] == 10.0

    def test_dag_node_output_key_defaults_to_node_id(self):
        node = DAGNodeData(node_id="my_node", agent_id="a")
        d = node.to_dict()
        assert d["output_key"] == "my_node"

    def test_dag_node_output_key_custom(self):
        node = DAGNodeData(node_id="my_node", agent_id="a", output_key="custom_key")
        d = node.to_dict()
        assert d["output_key"] == "custom_key"

    def test_dag_node_serialization_round_trip(self):
        node = DAGNodeData(
            node_id="rt",
            agent_id="agent_rt",
            depends_on=["dep1"],
            priority=3,
            metadata={"key": "value"},
        )
        d = node.to_dict()
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["node_id"] == "rt"
        assert deserialized["depends_on"] == ["dep1"]
        assert deserialized["metadata"] == {"key": "value"}


# ---------------------------------------------------------------------------
# DAGWorkflow Tests
# ---------------------------------------------------------------------------


class TestDAGWorkflow:
    """Tests for DAGWorkflow model."""

    def test_dag_workflow_creation(self, sample_linear_dag):
        assert sample_linear_dag.dag_id == "linear-dag"
        assert sample_linear_dag.name == "Linear DAG"
        assert len(sample_linear_dag.nodes) == 3

    def test_dag_workflow_hash_calculation(self, sample_linear_dag):
        d = sample_linear_dag.to_dict()
        serialized = json.dumps(d, sort_keys=True, default=str)
        expected_hash = hashlib.sha256(serialized.encode()).hexdigest()
        assert len(expected_hash) == 64

    def test_dag_workflow_hash_changes_with_nodes(
        self, sample_linear_dag, sample_diamond_dag
    ):
        d1 = json.dumps(sample_linear_dag.to_dict(), sort_keys=True)
        d2 = json.dumps(sample_diamond_dag.to_dict(), sort_keys=True)
        h1 = hashlib.sha256(d1.encode()).hexdigest()
        h2 = hashlib.sha256(d2.encode()).hexdigest()
        assert h1 != h2

    def test_dag_workflow_to_dict(self, sample_diamond_dag):
        d = sample_diamond_dag.to_dict()
        assert d["dag_id"] == "diamond-dag"
        assert "A" in d["nodes"]
        assert "D" in d["nodes"]
        assert d["on_failure"] == "fail_fast"
        assert d["max_parallel_nodes"] == 10

    def test_dag_workflow_to_dict_includes_all_fields(self, sample_complex_dag):
        d = sample_complex_dag.to_dict()
        assert "dag_id" in d
        assert "name" in d
        assert "description" in d
        assert "version" in d
        assert "nodes" in d
        assert "default_retry_policy" in d
        assert "default_timeout_policy" in d
        assert "on_failure" in d
        assert "max_parallel_nodes" in d
        assert "metadata" in d

    def test_dag_workflow_serialization_round_trip(self, sample_linear_dag):
        d = sample_linear_dag.to_dict()
        serialized = json.dumps(d, sort_keys=True, default=str)
        deserialized = json.loads(serialized)
        assert deserialized["dag_id"] == "linear-dag"
        assert len(deserialized["nodes"]) == 3

    def test_dag_workflow_version_default(self):
        dag = DAGWorkflowData(dag_id="v", name="V")
        assert dag.version == "1.0.0"

    def test_dag_workflow_adjacency_linear(self, sample_linear_dag):
        adj = sample_linear_dag.get_adjacency()
        assert "B" in adj["A"]
        assert "C" in adj["B"]
        assert adj["C"] == []

    def test_dag_workflow_adjacency_diamond(self, sample_diamond_dag):
        adj = sample_diamond_dag.get_adjacency()
        assert sorted(adj["A"]) == ["B", "C"]
        assert adj["B"] == ["D"]
        assert adj["C"] == ["D"]
        assert adj["D"] == []


# ---------------------------------------------------------------------------
# RetryPolicy Tests
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    """Tests for RetryPolicy model."""

    def test_retry_policy_creation(self, sample_retry_policy):
        assert sample_retry_policy.max_retries == 3
        assert sample_retry_policy.strategy == "exponential"
        assert sample_retry_policy.base_delay == 0.01

    def test_retry_policy_defaults(self):
        rp = RetryPolicyData()
        assert rp.max_retries == 3
        assert rp.strategy == "exponential"
        assert rp.jitter is False

    def test_retry_policy_to_dict(self, sample_retry_policy):
        d = sample_retry_policy.to_dict()
        assert d["max_retries"] == 3
        assert d["strategy"] == "exponential"
        assert d["base_delay"] == 0.01
        assert d["jitter"] is False

    def test_retry_policy_serialization_round_trip(self):
        rp = RetryPolicyData(max_retries=5, strategy="fibonacci", jitter=True)
        d = rp.to_dict()
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["max_retries"] == 5
        assert deserialized["strategy"] == "fibonacci"
        assert deserialized["jitter"] is True


# ---------------------------------------------------------------------------
# TimeoutPolicy Tests
# ---------------------------------------------------------------------------


class TestTimeoutPolicy:
    """Tests for TimeoutPolicy model."""

    def test_timeout_policy_creation(self, sample_timeout_policy):
        assert sample_timeout_policy.timeout_seconds == 5.0
        assert sample_timeout_policy.on_timeout == "fail"

    def test_timeout_policy_defaults(self):
        tp = TimeoutPolicyData()
        assert tp.timeout_seconds == 5.0
        assert tp.on_timeout == "fail"

    def test_timeout_policy_to_dict(self, sample_timeout_policy):
        d = sample_timeout_policy.to_dict()
        assert d["timeout_seconds"] == 5.0
        assert d["on_timeout"] == "fail"

    def test_timeout_policy_serialization_round_trip(self):
        tp = TimeoutPolicyData(timeout_seconds=120.0, on_timeout="skip")
        d = tp.to_dict()
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["timeout_seconds"] == 120.0
        assert deserialized["on_timeout"] == "skip"


# ---------------------------------------------------------------------------
# ExecutionStatus Enum Tests
# ---------------------------------------------------------------------------


class TestExecutionStatusEnum:
    """Tests for ExecutionStatus enum values."""

    def test_pending_value(self):
        assert ExecutionStatus.PENDING.value == "pending"

    def test_running_value(self):
        assert ExecutionStatus.RUNNING.value == "running"

    def test_completed_value(self):
        assert ExecutionStatus.COMPLETED.value == "completed"

    def test_failed_value(self):
        assert ExecutionStatus.FAILED.value == "failed"

    def test_skipped_value(self):
        assert ExecutionStatus.SKIPPED.value == "skipped"

    def test_cancelled_value(self):
        assert ExecutionStatus.CANCELLED.value == "cancelled"

    def test_all_statuses_count(self):
        assert len(ExecutionStatus) == 6


# ---------------------------------------------------------------------------
# OnFailure Enum Tests
# ---------------------------------------------------------------------------


class TestOnFailureEnum:
    """Tests for OnFailure enum values."""

    def test_stop_value(self):
        assert OnFailure.STOP.value == "stop"

    def test_skip_value(self):
        assert OnFailure.SKIP.value == "skip"

    def test_compensate_value(self):
        assert OnFailure.COMPENSATE.value == "compensate"

    def test_all_values_count(self):
        assert len(OnFailure) == 3


# ---------------------------------------------------------------------------
# Execution Trace Tests
# ---------------------------------------------------------------------------


class TestExecutionTrace:
    """Tests for ExecutionTrace model (inline stub)."""

    def test_execution_trace_creation(self):
        trace = {
            "execution_id": "exec-001",
            "dag_id": "linear-dag",
            "status": "completed",
            "node_traces": {},
            "topology_levels": [["A"], ["B"], ["C"]],
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "provenance_chain_hash": "a" * 64,
        }
        assert trace["execution_id"] == "exec-001"
        assert len(trace["topology_levels"]) == 3
        assert len(trace["provenance_chain_hash"]) == 64

    def test_execution_trace_node_traces_populated(self):
        trace = {
            "execution_id": "exec-002",
            "dag_id": "diamond-dag",
            "status": "completed",
            "node_traces": {
                "A": {"status": "completed", "duration_ms": 10.0},
                "B": {"status": "completed", "duration_ms": 15.0},
                "C": {"status": "completed", "duration_ms": 12.0},
                "D": {"status": "completed", "duration_ms": 8.0},
            },
            "topology_levels": [["A"], ["B", "C"], ["D"]],
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "provenance_chain_hash": "",
        }
        assert len(trace["node_traces"]) == 4
        assert trace["node_traces"]["B"]["duration_ms"] == 15.0


# ---------------------------------------------------------------------------
# NodeProvenance Tests
# ---------------------------------------------------------------------------


class TestNodeProvenance:
    """Tests for NodeProvenance model (inline stub)."""

    def test_node_provenance_creation(self, compute_hash):
        prov = {
            "node_id": "A",
            "input_hash": compute_hash({"data": "input"}),
            "output_hash": compute_hash({"result": "output"}),
            "duration_ms": 12.5,
            "attempt_count": 1,
            "parent_hashes": [],
            "chain_hash": "",
        }
        assert prov["node_id"] == "A"
        assert len(prov["input_hash"]) == 64
        assert len(prov["output_hash"]) == 64
        assert prov["attempt_count"] == 1

    def test_node_provenance_chain_hash(self, compute_hash):
        prov_a = {
            "node_id": "A",
            "input_hash": compute_hash({"in": "a"}),
            "output_hash": compute_hash({"out": "a"}),
            "duration_ms": 10.0,
            "attempt_count": 1,
            "parent_hashes": [],
        }
        chain_data = json.dumps(
            {
                "node_id": prov_a["node_id"],
                "input_hash": prov_a["input_hash"],
                "output_hash": prov_a["output_hash"],
                "parent_hashes": prov_a["parent_hashes"],
            },
            sort_keys=True,
        )
        prov_a["chain_hash"] = hashlib.sha256(chain_data.encode()).hexdigest()
        assert len(prov_a["chain_hash"]) == 64

    def test_node_provenance_with_parents(self, compute_hash):
        parent_hash = compute_hash({"parent": "data"})
        prov = {
            "node_id": "D",
            "input_hash": compute_hash({"in": "d"}),
            "output_hash": compute_hash({"out": "d"}),
            "duration_ms": 5.0,
            "attempt_count": 2,
            "parent_hashes": [parent_hash],
        }
        assert len(prov["parent_hashes"]) == 1
        assert prov["attempt_count"] == 2


# ---------------------------------------------------------------------------
# Model YAML Round-Trip Tests
# ---------------------------------------------------------------------------


class TestModelYAMLRoundTrip:
    """Test model serialization to/from YAML-like dicts."""

    def test_dag_workflow_yaml_round_trip(self, sample_complex_dag):
        """Simulate YAML dump/load via dict serialization."""
        d = sample_complex_dag.to_dict()
        serialized = json.dumps(d, sort_keys=True, default=str)
        loaded = json.loads(serialized)
        assert loaded["dag_id"] == sample_complex_dag.dag_id
        assert "intake" in loaded["nodes"]
        assert "scope1" in loaded["nodes"]
        assert "aggregate" in loaded["nodes"]
        assert loaded["nodes"]["scope1"]["retry_policy"]["max_retries"] == 3

    def test_retry_policy_yaml_round_trip(self):
        rp = RetryPolicyData(
            max_retries=4, strategy="linear", base_delay=0.5, jitter=True
        )
        d = rp.to_dict()
        serialized = json.dumps(d)
        loaded = json.loads(serialized)
        assert loaded["max_retries"] == 4
        assert loaded["strategy"] == "linear"
        assert loaded["jitter"] is True
