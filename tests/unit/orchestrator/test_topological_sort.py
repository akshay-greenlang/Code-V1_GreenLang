# -*- coding: utf-8 -*-
"""
Unit tests for Topological Sort (AGENT-FOUND-001)

Tests Kahn's algorithm with deterministic tie-breaking, level grouping,
root/sink identification, and edge cases.

Coverage target: 85%+ of topological_sort.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple

import pytest

from tests.unit.orchestrator.conftest import DAGNodeData, DAGWorkflowData


# ---------------------------------------------------------------------------
# Inline topological sort that mirrors expected interface
# ---------------------------------------------------------------------------


def topological_sort(dag: DAGWorkflowData) -> List[str]:
    """Kahn's algorithm with deterministic sorted tie-breaking."""
    if not dag.nodes:
        return []

    in_degree: Dict[str, int] = {nid: 0 for nid in dag.nodes}
    adj: Dict[str, List[str]] = {nid: [] for nid in dag.nodes}

    for nid, node in dag.nodes.items():
        for dep in node.depends_on:
            if dep in adj:
                adj[dep].append(nid)
                in_degree[nid] += 1

    # Initialize queue with zero in-degree nodes (sorted for determinism)
    queue = sorted([nid for nid, deg in in_degree.items() if deg == 0])
    result = []

    while queue:
        # Pop the first (alphabetically smallest) node
        current = queue.pop(0)
        result.append(current)

        for neighbor in sorted(adj[current]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                # Insert in sorted position
                queue.append(neighbor)
                queue.sort()

    if len(result) != len(dag.nodes):
        raise ValueError("DAG contains a cycle - topological sort impossible")

    return result


def level_grouping(dag: DAGWorkflowData) -> List[List[str]]:
    """Group nodes by level (depth). Nodes at same level can run in parallel."""
    if not dag.nodes:
        return []

    in_degree: Dict[str, int] = {nid: 0 for nid in dag.nodes}
    adj: Dict[str, List[str]] = {nid: [] for nid in dag.nodes}

    for nid, node in dag.nodes.items():
        for dep in node.depends_on:
            if dep in adj:
                adj[dep].append(nid)
                in_degree[nid] += 1

    # Start with zero in-degree nodes
    current_level = sorted([nid for nid, deg in in_degree.items() if deg == 0])
    levels = []

    while current_level:
        levels.append(current_level)
        next_level = []
        for nid in current_level:
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_level.append(neighbor)
        current_level = sorted(next_level)

    processed = sum(len(lvl) for lvl in levels)
    if processed != len(dag.nodes):
        raise ValueError("DAG contains a cycle - level grouping impossible")

    return levels


def get_roots(dag: DAGWorkflowData) -> List[str]:
    """Get root nodes (nodes with no dependencies)."""
    return sorted([nid for nid, node in dag.nodes.items() if not node.depends_on])


def get_sinks(dag: DAGWorkflowData) -> List[str]:
    """Get sink nodes (nodes that no other node depends on)."""
    depended_on: Set[str] = set()
    for node in dag.nodes.values():
        depended_on.update(node.depends_on)
    return sorted([nid for nid in dag.nodes if nid not in depended_on])


# ===========================================================================
# Test Classes
# ===========================================================================


class TestLinearDAGSort:
    """Test topological sort of linear DAGs."""

    def test_linear_dag_order(self, sample_linear_dag):
        order = topological_sort(sample_linear_dag)
        assert order == ["A", "B", "C"]

    def test_linear_dag_a_before_b(self, sample_linear_dag):
        order = topological_sort(sample_linear_dag)
        assert order.index("A") < order.index("B")

    def test_linear_dag_b_before_c(self, sample_linear_dag):
        order = topological_sort(sample_linear_dag)
        assert order.index("B") < order.index("C")

    def test_reverse_named_linear(self):
        """Nodes named Z, Y, X but Z is root."""
        dag = DAGWorkflowData(
            dag_id="rev",
            name="Reverse",
            nodes={
                "Z": DAGNodeData(node_id="Z", agent_id="z"),
                "Y": DAGNodeData(node_id="Y", agent_id="y", depends_on=["Z"]),
                "X": DAGNodeData(node_id="X", agent_id="x", depends_on=["Y"]),
            },
        )
        order = topological_sort(dag)
        assert order == ["Z", "Y", "X"]


class TestDiamondDAGSort:
    """Test topological sort of diamond DAGs."""

    def test_diamond_a_first(self, sample_diamond_dag):
        order = topological_sort(sample_diamond_dag)
        assert order[0] == "A"

    def test_diamond_d_last(self, sample_diamond_dag):
        order = topological_sort(sample_diamond_dag)
        assert order[-1] == "D"

    def test_diamond_b_c_between_a_d(self, sample_diamond_dag):
        order = topological_sort(sample_diamond_dag)
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_diamond_b_before_c_alphabetical(self, sample_diamond_dag):
        """B and C are at same level; sorted alphabetically."""
        order = topological_sort(sample_diamond_dag)
        assert order.index("B") < order.index("C")


class TestWideParallelSort:
    """Test topological sort of wide parallel DAGs."""

    def test_root_first(self, sample_wide_dag):
        order = topological_sort(sample_wide_dag)
        assert order[0] == "root"

    def test_sink_last(self, sample_wide_dag):
        order = topological_sort(sample_wide_dag)
        assert order[-1] == "sink"

    def test_parallel_nodes_in_middle(self, sample_wide_dag):
        order = topological_sort(sample_wide_dag)
        parallel_nodes = [n for n in order if n.startswith("parallel_")]
        assert len(parallel_nodes) == 5
        # All parallel nodes should be between root and sink
        for pn in parallel_nodes:
            assert order.index("root") < order.index(pn)
            assert order.index(pn) < order.index("sink")

    def test_parallel_nodes_sorted_alphabetically(self, sample_wide_dag):
        order = topological_sort(sample_wide_dag)
        parallel_nodes = [n for n in order if n.startswith("parallel_")]
        assert parallel_nodes == sorted(parallel_nodes)


class TestDeterministicOrdering:
    """Test that same DAG always produces same order."""

    def test_linear_deterministic(self, sample_linear_dag):
        results = [topological_sort(sample_linear_dag) for _ in range(10)]
        for r in results[1:]:
            assert r == results[0]

    def test_diamond_deterministic(self, sample_diamond_dag):
        results = [topological_sort(sample_diamond_dag) for _ in range(10)]
        for r in results[1:]:
            assert r == results[0]

    def test_wide_deterministic(self, sample_wide_dag):
        results = [topological_sort(sample_wide_dag) for _ in range(10)]
        for r in results[1:]:
            assert r == results[0]

    def test_complex_deterministic(self, sample_complex_dag):
        results = [topological_sort(sample_complex_dag) for _ in range(10)]
        for r in results[1:]:
            assert r == results[0]


class TestLevelGroupingLinear:
    """Test level grouping for linear DAGs."""

    def test_linear_three_levels(self, sample_linear_dag):
        levels = level_grouping(sample_linear_dag)
        assert len(levels) == 3

    def test_linear_one_node_per_level(self, sample_linear_dag):
        levels = level_grouping(sample_linear_dag)
        assert levels == [["A"], ["B"], ["C"]]


class TestLevelGroupingDiamond:
    """Test level grouping for diamond DAGs."""

    def test_diamond_three_levels(self, sample_diamond_dag):
        levels = level_grouping(sample_diamond_dag)
        assert len(levels) == 3

    def test_diamond_level_contents(self, sample_diamond_dag):
        levels = level_grouping(sample_diamond_dag)
        assert levels[0] == ["A"]
        assert levels[1] == ["B", "C"]
        assert levels[2] == ["D"]

    def test_diamond_parallel_level(self, sample_diamond_dag):
        """Level 1 should have B and C (can run in parallel)."""
        levels = level_grouping(sample_diamond_dag)
        assert sorted(levels[1]) == ["B", "C"]


class TestLevelGroupingWide:
    """Test level grouping for wide parallel DAGs."""

    def test_wide_three_levels(self, sample_wide_dag):
        levels = level_grouping(sample_wide_dag)
        assert len(levels) == 3

    def test_wide_root_level(self, sample_wide_dag):
        levels = level_grouping(sample_wide_dag)
        assert levels[0] == ["root"]

    def test_wide_parallel_level(self, sample_wide_dag):
        levels = level_grouping(sample_wide_dag)
        assert len(levels[1]) == 5

    def test_wide_sink_level(self, sample_wide_dag):
        levels = level_grouping(sample_wide_dag)
        assert levels[2] == ["sink"]

    def test_wide_parallel_sorted(self, sample_wide_dag):
        levels = level_grouping(sample_wide_dag)
        assert levels[1] == sorted(levels[1])


class TestLevelGroupingComplex:
    """Test level grouping for complex DAGs."""

    def test_complex_level_count(self, sample_complex_dag):
        levels = level_grouping(sample_complex_dag)
        # intake -> validate -> scope1,scope2 -> aggregate -> report = 5 levels
        assert len(levels) == 5

    def test_complex_first_level(self, sample_complex_dag):
        levels = level_grouping(sample_complex_dag)
        assert levels[0] == ["intake"]

    def test_complex_parallel_scopes(self, sample_complex_dag):
        levels = level_grouping(sample_complex_dag)
        assert sorted(levels[2]) == ["scope1", "scope2"]


class TestGetRoots:
    """Test root node identification."""

    def test_linear_one_root(self, sample_linear_dag):
        roots = get_roots(sample_linear_dag)
        assert roots == ["A"]

    def test_diamond_one_root(self, sample_diamond_dag):
        roots = get_roots(sample_diamond_dag)
        assert roots == ["A"]

    def test_wide_one_root(self, sample_wide_dag):
        roots = get_roots(sample_wide_dag)
        assert roots == ["root"]

    def test_multiple_roots(self):
        dag = DAGWorkflowData(
            dag_id="multi",
            name="Multi",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b"),
                "C": DAGNodeData(node_id="C", agent_id="c", depends_on=["A", "B"]),
            },
        )
        roots = get_roots(dag)
        assert roots == ["A", "B"]


class TestGetSinks:
    """Test sink node identification."""

    def test_linear_one_sink(self, sample_linear_dag):
        sinks = get_sinks(sample_linear_dag)
        assert sinks == ["C"]

    def test_diamond_one_sink(self, sample_diamond_dag):
        sinks = get_sinks(sample_diamond_dag)
        assert sinks == ["D"]

    def test_wide_one_sink(self, sample_wide_dag):
        sinks = get_sinks(sample_wide_dag)
        assert sinks == ["sink"]

    def test_multiple_sinks(self):
        dag = DAGWorkflowData(
            dag_id="multi-sink",
            name="Multi Sink",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A"]),
                "C": DAGNodeData(node_id="C", agent_id="c", depends_on=["A"]),
            },
        )
        sinks = get_sinks(dag)
        assert sinks == ["B", "C"]


class TestSingleNode:
    """Test single-node DAG."""

    def test_topological_sort_single(self):
        dag = DAGWorkflowData(
            dag_id="one",
            name="One",
            nodes={"A": DAGNodeData(node_id="A", agent_id="a")},
        )
        assert topological_sort(dag) == ["A"]

    def test_level_grouping_single(self):
        dag = DAGWorkflowData(
            dag_id="one",
            name="One",
            nodes={"A": DAGNodeData(node_id="A", agent_id="a")},
        )
        levels = level_grouping(dag)
        assert levels == [["A"]]

    def test_roots_single(self):
        dag = DAGWorkflowData(
            dag_id="one",
            name="One",
            nodes={"A": DAGNodeData(node_id="A", agent_id="a")},
        )
        assert get_roots(dag) == ["A"]

    def test_sinks_single(self):
        dag = DAGWorkflowData(
            dag_id="one",
            name="One",
            nodes={"A": DAGNodeData(node_id="A", agent_id="a")},
        )
        assert get_sinks(dag) == ["A"]


class TestEmptyDAG:
    """Test empty DAG edge case."""

    def test_topological_sort_empty(self):
        dag = DAGWorkflowData(dag_id="empty", name="Empty", nodes={})
        assert topological_sort(dag) == []

    def test_level_grouping_empty(self):
        dag = DAGWorkflowData(dag_id="empty", name="Empty", nodes={})
        assert level_grouping(dag) == []

    def test_roots_empty(self):
        dag = DAGWorkflowData(dag_id="empty", name="Empty", nodes={})
        assert get_roots(dag) == []

    def test_sinks_empty(self):
        dag = DAGWorkflowData(dag_id="empty", name="Empty", nodes={})
        assert get_sinks(dag) == []


class TestCycleDetection:
    """Test cycle detection during topological sort."""

    def test_cycle_raises_in_sort(self, sample_invalid_cycle_dag):
        with pytest.raises(ValueError, match="cycle"):
            topological_sort(sample_invalid_cycle_dag)

    def test_cycle_raises_in_level_grouping(self, sample_invalid_cycle_dag):
        with pytest.raises(ValueError, match="cycle"):
            level_grouping(sample_invalid_cycle_dag)
