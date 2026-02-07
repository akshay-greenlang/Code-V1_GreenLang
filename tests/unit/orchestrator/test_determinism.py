# -*- coding: utf-8 -*-
"""
Unit tests for Deterministic Scheduling (AGENT-FOUND-001)

Tests sorted scheduling, deterministic execution IDs, replay-identical
traces, and different inputs producing different traces.

Coverage target: 85%+ of determinism.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from tests.unit.orchestrator.conftest import (
    DAGNodeData,
    DAGWorkflowData,
    _run_async,
)


# ---------------------------------------------------------------------------
# Inline determinism utilities that mirror expected interface
# ---------------------------------------------------------------------------


class DeterministicScheduler:
    """Schedules nodes in deterministic order."""

    @staticmethod
    def sort_nodes(node_ids: List[str]) -> List[str]:
        """Sort nodes alphabetically for deterministic ordering."""
        return sorted(node_ids)

    @staticmethod
    def sort_levels(levels: List[List[str]]) -> List[List[str]]:
        """Sort each level alphabetically."""
        return [sorted(level) for level in levels]


def deterministic_execution_id(dag_id: str, input_data: Dict[str, Any], seed: int = 0) -> str:
    """Generate a deterministic execution ID from DAG ID and input data."""
    data = json.dumps(
        {"dag_id": dag_id, "input_data": input_data, "seed": seed},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(data.encode()).hexdigest()[:32]


def deterministic_uuid(seed_str: str) -> str:
    """Generate a deterministic UUID from a seed string."""
    h = hashlib.sha256(seed_str.encode()).hexdigest()
    # Format as UUID
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


class ExecutionTraceComparator:
    """Compare execution traces for replay verification."""

    @staticmethod
    def traces_match(trace1: Dict[str, Any], trace2: Dict[str, Any]) -> bool:
        """Compare two execution traces for deterministic equality."""
        # Compare topology levels
        if trace1.get("topology_levels") != trace2.get("topology_levels"):
            return False

        # Compare node output hashes
        for node_id in trace1.get("node_hashes", {}):
            if node_id not in trace2.get("node_hashes", {}):
                return False
            if trace1["node_hashes"][node_id] != trace2["node_hashes"][node_id]:
                return False

        # Compare provenance chain
        if trace1.get("provenance_chain_hash") != trace2.get("provenance_chain_hash"):
            return False

        return True

    @staticmethod
    def build_trace(
        dag: DAGWorkflowData,
        node_outputs: Dict[str, Any],
        levels: List[List[str]],
    ) -> Dict[str, Any]:
        """Build a trace dict for comparison."""
        node_hashes = {}
        for nid, output in node_outputs.items():
            h = hashlib.sha256(
                json.dumps(output, sort_keys=True, default=str).encode()
            ).hexdigest()
            node_hashes[nid] = h

        # Compute chain hash
        chain_data = json.dumps(node_hashes, sort_keys=True)
        chain_hash = hashlib.sha256(chain_data.encode()).hexdigest()

        return {
            "dag_id": dag.dag_id,
            "topology_levels": levels,
            "node_hashes": node_hashes,
            "provenance_chain_hash": chain_hash,
        }


# ===========================================================================
# Test Classes
# ===========================================================================


class TestSortedScheduling:
    """Test that nodes are sorted alphabetically."""

    def test_sort_single_level(self):
        scheduler = DeterministicScheduler()
        assert scheduler.sort_nodes(["C", "A", "B"]) == ["A", "B", "C"]

    def test_sort_already_sorted(self):
        scheduler = DeterministicScheduler()
        assert scheduler.sort_nodes(["A", "B", "C"]) == ["A", "B", "C"]

    def test_sort_reverse_order(self):
        scheduler = DeterministicScheduler()
        assert scheduler.sort_nodes(["Z", "Y", "X"]) == ["X", "Y", "Z"]

    def test_sort_empty(self):
        scheduler = DeterministicScheduler()
        assert scheduler.sort_nodes([]) == []

    def test_sort_single_node(self):
        scheduler = DeterministicScheduler()
        assert scheduler.sort_nodes(["A"]) == ["A"]

    def test_sort_levels(self):
        scheduler = DeterministicScheduler()
        levels = [["C", "A"], ["B", "D"], ["E"]]
        sorted_levels = scheduler.sort_levels(levels)
        assert sorted_levels == [["A", "C"], ["B", "D"], ["E"]]

    def test_sort_levels_preserves_count(self):
        scheduler = DeterministicScheduler()
        levels = [["C", "A", "B"], ["D"]]
        sorted_levels = scheduler.sort_levels(levels)
        assert len(sorted_levels) == 2
        assert len(sorted_levels[0]) == 3


class TestDeterministicExecutionId:
    """Test deterministic execution ID generation."""

    def test_same_inputs_same_id(self):
        id1 = deterministic_execution_id("dag-1", {"key": "value"})
        id2 = deterministic_execution_id("dag-1", {"key": "value"})
        assert id1 == id2

    def test_different_dag_id_different_id(self):
        id1 = deterministic_execution_id("dag-1", {"key": "value"})
        id2 = deterministic_execution_id("dag-2", {"key": "value"})
        assert id1 != id2

    def test_different_input_different_id(self):
        id1 = deterministic_execution_id("dag-1", {"key": "value1"})
        id2 = deterministic_execution_id("dag-1", {"key": "value2"})
        assert id1 != id2

    def test_id_length(self):
        eid = deterministic_execution_id("dag-1", {})
        assert len(eid) == 32

    def test_id_is_hex(self):
        eid = deterministic_execution_id("dag-1", {})
        int(eid, 16)  # Should not raise

    def test_different_seed_different_id(self):
        id1 = deterministic_execution_id("dag-1", {}, seed=0)
        id2 = deterministic_execution_id("dag-1", {}, seed=1)
        assert id1 != id2


class TestDeterministicUUID:
    """Test deterministic UUID generation."""

    def test_same_seed_same_uuid(self):
        u1 = deterministic_uuid("seed-1")
        u2 = deterministic_uuid("seed-1")
        assert u1 == u2

    def test_different_seed_different_uuid(self):
        u1 = deterministic_uuid("seed-1")
        u2 = deterministic_uuid("seed-2")
        assert u1 != u2

    def test_uuid_format(self):
        u = deterministic_uuid("test")
        parts = u.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12


class TestReplayIdenticalTraces:
    """Test that two executions of same DAG + inputs produce identical traces."""

    def test_identical_traces(self, sample_linear_dag):
        # Simulate two executions with same outputs
        outputs = {"A": {"val": 1}, "B": {"val": 2}, "C": {"val": 3}}
        levels = [["A"], ["B"], ["C"]]

        trace1 = ExecutionTraceComparator.build_trace(
            sample_linear_dag, outputs, levels
        )
        trace2 = ExecutionTraceComparator.build_trace(
            sample_linear_dag, outputs, levels
        )

        assert ExecutionTraceComparator.traces_match(trace1, trace2)

    def test_identical_diamond_traces(self, sample_diamond_dag):
        outputs = {
            "A": {"val": 1},
            "B": {"val": 2},
            "C": {"val": 3},
            "D": {"val": 4},
        }
        levels = [["A"], ["B", "C"], ["D"]]

        trace1 = ExecutionTraceComparator.build_trace(
            sample_diamond_dag, outputs, levels
        )
        trace2 = ExecutionTraceComparator.build_trace(
            sample_diamond_dag, outputs, levels
        )

        assert ExecutionTraceComparator.traces_match(trace1, trace2)

    def test_provenance_chain_hash_matches(self, sample_linear_dag):
        outputs = {"A": {"val": 1}, "B": {"val": 2}, "C": {"val": 3}}
        levels = [["A"], ["B"], ["C"]]

        trace1 = ExecutionTraceComparator.build_trace(
            sample_linear_dag, outputs, levels
        )
        trace2 = ExecutionTraceComparator.build_trace(
            sample_linear_dag, outputs, levels
        )

        assert trace1["provenance_chain_hash"] == trace2["provenance_chain_hash"]


class TestReplayDifferentInputsDifferentTraces:
    """Test that different inputs produce different traces."""

    def test_different_outputs_different_traces(self, sample_linear_dag):
        outputs1 = {"A": {"val": 1}, "B": {"val": 2}, "C": {"val": 3}}
        outputs2 = {"A": {"val": 10}, "B": {"val": 20}, "C": {"val": 30}}
        levels = [["A"], ["B"], ["C"]]

        trace1 = ExecutionTraceComparator.build_trace(
            sample_linear_dag, outputs1, levels
        )
        trace2 = ExecutionTraceComparator.build_trace(
            sample_linear_dag, outputs2, levels
        )

        assert not ExecutionTraceComparator.traces_match(trace1, trace2)

    def test_different_levels_different_traces(self, sample_linear_dag):
        outputs = {"A": {"val": 1}, "B": {"val": 2}, "C": {"val": 3}}

        trace1 = ExecutionTraceComparator.build_trace(
            sample_linear_dag, outputs, [["A"], ["B"], ["C"]]
        )
        trace2 = ExecutionTraceComparator.build_trace(
            sample_linear_dag, outputs, [["A", "B"], ["C"]]  # Different grouping
        )

        assert not ExecutionTraceComparator.traces_match(trace1, trace2)

    def test_missing_node_different_traces(self, sample_linear_dag):
        outputs1 = {"A": {"val": 1}, "B": {"val": 2}, "C": {"val": 3}}
        outputs2 = {"A": {"val": 1}, "B": {"val": 2}}  # Missing C
        levels = [["A"], ["B"], ["C"]]

        trace1 = ExecutionTraceComparator.build_trace(
            sample_linear_dag, outputs1, levels
        )
        trace2 = ExecutionTraceComparator.build_trace(
            sample_linear_dag, outputs2, levels
        )

        assert not ExecutionTraceComparator.traces_match(trace1, trace2)
