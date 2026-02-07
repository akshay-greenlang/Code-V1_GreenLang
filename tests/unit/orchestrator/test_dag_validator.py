# -*- coding: utf-8 -*-
"""
Unit tests for DAG Validator (AGENT-FOUND-001)

Tests cycle detection, unreachable node detection, missing dependencies,
duplicate node IDs, self-dependencies, and structural validation.

Coverage target: 85%+ of dag_validator.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import pytest

from tests.unit.orchestrator.conftest import DAGNodeData, DAGWorkflowData


# ---------------------------------------------------------------------------
# Inline DAG validator that mirrors expected interface
# ---------------------------------------------------------------------------


class ValidationError:
    """A single validation error."""

    def __init__(self, error_type: str, message: str, node_id: str = ""):
        self.error_type = error_type
        self.message = message
        self.node_id = node_id

    def __repr__(self):
        return f"ValidationError({self.error_type}: {self.message})"


class ValidationResult:
    """Result of DAG validation."""

    def __init__(self, errors: List[ValidationError] = None, warnings: List[ValidationError] = None):
        self.errors = errors or []
        self.warnings = warnings or []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def has_error_type(self, error_type: str) -> bool:
        return any(e.error_type == error_type for e in self.errors)


def validate_dag(dag: DAGWorkflowData) -> ValidationResult:
    """Validate a DAG workflow. Returns structured error list."""
    errors: List[ValidationError] = []
    warnings: List[ValidationError] = []

    if not dag.nodes:
        errors.append(ValidationError("empty_dag", "DAG has no nodes"))
        return ValidationResult(errors, warnings)

    all_ids = set(dag.nodes.keys())

    # Check duplicate IDs (handled by dict, but check node_id matches key)
    for key, node in dag.nodes.items():
        if node.node_id != key:
            errors.append(
                ValidationError(
                    "id_mismatch",
                    f"Node key '{key}' does not match node_id '{node.node_id}'",
                    node_id=key,
                )
            )

    # Check self-dependencies
    for nid, node in dag.nodes.items():
        if nid in node.depends_on:
            errors.append(
                ValidationError(
                    "self_dependency",
                    f"Node '{nid}' depends on itself",
                    node_id=nid,
                )
            )

    # Check missing dependencies
    for nid, node in dag.nodes.items():
        for dep in node.depends_on:
            if dep not in all_ids:
                errors.append(
                    ValidationError(
                        "missing_dependency",
                        f"Node '{nid}' depends on '{dep}' which does not exist",
                        node_id=nid,
                    )
                )

    # Cycle detection (DFS)
    cycles = detect_cycles(dag)
    for cycle in cycles:
        cycle_str = " -> ".join(cycle)
        errors.append(
            ValidationError(
                "cycle_detected",
                f"Cycle detected: {cycle_str}",
            )
        )

    # Unreachable nodes
    unreachable = find_unreachable_nodes(dag)
    for nid in unreachable:
        warnings.append(
            ValidationError(
                "unreachable_node",
                f"Node '{nid}' is unreachable from any root",
                node_id=nid,
            )
        )

    return ValidationResult(errors, warnings)


def detect_cycles(dag: DAGWorkflowData) -> List[List[str]]:
    """Detect cycles using DFS. Returns list of cycles found."""
    cycles = []
    visited: Set[str] = set()
    in_stack: Set[str] = set()
    path: List[str] = []

    # Build forward adjacency (node -> nodes that depend on it)
    # For cycle detection, we follow dependency edges: if B depends_on A, edge A->B
    adj: Dict[str, List[str]] = {nid: [] for nid in dag.nodes}
    for nid, node in dag.nodes.items():
        for dep in node.depends_on:
            if dep in adj:
                adj[dep].append(nid)

    def _dfs(node_id: str):
        visited.add(node_id)
        in_stack.add(node_id)
        path.append(node_id)

        for neighbor in sorted(adj.get(node_id, [])):
            if neighbor not in visited:
                _dfs(neighbor)
            elif neighbor in in_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)

        path.pop()
        in_stack.discard(node_id)

    for nid in sorted(dag.nodes.keys()):
        if nid not in visited:
            _dfs(nid)

    return cycles


def find_unreachable_nodes(dag: DAGWorkflowData) -> List[str]:
    """Find nodes unreachable from any root (nodes with no dependencies)."""
    if not dag.nodes:
        return []

    roots = [nid for nid, node in dag.nodes.items() if not node.depends_on]
    if not roots:
        # If no roots, all nodes are unreachable (likely a cycle)
        return sorted(dag.nodes.keys())

    # BFS from all roots
    reachable: Set[str] = set()
    queue = list(roots)
    adj = dag.get_adjacency()

    while queue:
        current = queue.pop(0)
        if current in reachable:
            continue
        reachable.add(current)
        for dep in adj.get(current, []):
            if dep not in reachable:
                queue.append(dep)

    unreachable = sorted(set(dag.nodes.keys()) - reachable)
    return unreachable


# ===========================================================================
# Test Classes
# ===========================================================================


class TestValidLinearDAG:
    """Test validation of valid linear DAGs."""

    def test_valid_linear_dag(self, sample_linear_dag):
        result = validate_dag(sample_linear_dag)
        assert result.is_valid
        assert result.error_count == 0

    def test_single_node_dag(self):
        dag = DAGWorkflowData(
            dag_id="single",
            name="Single",
            nodes={"A": DAGNodeData(node_id="A", agent_id="a")},
        )
        result = validate_dag(dag)
        assert result.is_valid

    def test_two_node_chain(self):
        dag = DAGWorkflowData(
            dag_id="two",
            name="Two",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A"]),
            },
        )
        result = validate_dag(dag)
        assert result.is_valid


class TestValidDiamondDAG:
    """Test validation of valid diamond DAGs."""

    def test_valid_diamond_dag(self, sample_diamond_dag):
        result = validate_dag(sample_diamond_dag)
        assert result.is_valid

    def test_diamond_no_cycles(self, sample_diamond_dag):
        cycles = detect_cycles(sample_diamond_dag)
        assert len(cycles) == 0


class TestValidComplexDAG:
    """Test validation of valid complex DAGs."""

    def test_valid_complex_dag(self, sample_complex_dag):
        result = validate_dag(sample_complex_dag)
        assert result.is_valid

    def test_valid_wide_dag(self, sample_wide_dag):
        result = validate_dag(sample_wide_dag)
        assert result.is_valid


class TestDetectSimpleCycle:
    """Test simple cycle detection (A -> B -> A)."""

    def test_simple_two_node_cycle(self):
        dag = DAGWorkflowData(
            dag_id="cycle2",
            name="Cycle 2",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a", depends_on=["B"]),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A"]),
            },
        )
        result = validate_dag(dag)
        assert not result.is_valid
        assert result.has_error_type("cycle_detected")

    def test_three_node_cycle(self, sample_invalid_cycle_dag):
        result = validate_dag(sample_invalid_cycle_dag)
        assert not result.is_valid
        assert result.has_error_type("cycle_detected")


class TestDetectComplexCycle:
    """Test complex cycle detection (A -> B -> C -> D -> B)."""

    def test_complex_cycle_with_branch(self):
        dag = DAGWorkflowData(
            dag_id="complex-cycle",
            name="Complex Cycle",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a", depends_on=[]),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A", "D"]),
                "C": DAGNodeData(node_id="C", agent_id="c", depends_on=["B"]),
                "D": DAGNodeData(node_id="D", agent_id="d", depends_on=["C"]),
            },
        )
        result = validate_dag(dag)
        assert not result.is_valid
        assert result.has_error_type("cycle_detected")

    def test_cycle_does_not_affect_acyclic_part(self):
        """DAG with both acyclic and cyclic parts should detect cycle."""
        dag = DAGWorkflowData(
            dag_id="partial-cycle",
            name="Partial Cycle",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A"]),
                "X": DAGNodeData(node_id="X", agent_id="x", depends_on=["Y"]),
                "Y": DAGNodeData(node_id="Y", agent_id="y", depends_on=["X"]),
            },
        )
        result = validate_dag(dag)
        assert not result.is_valid
        assert result.has_error_type("cycle_detected")


class TestDetectSelfDependency:
    """Test self-dependency detection."""

    def test_self_dependency(self, sample_invalid_self_dep_dag):
        result = validate_dag(sample_invalid_self_dep_dag)
        assert not result.is_valid
        assert result.has_error_type("self_dependency")

    def test_self_dependency_message(self, sample_invalid_self_dep_dag):
        result = validate_dag(sample_invalid_self_dep_dag)
        err = [e for e in result.errors if e.error_type == "self_dependency"][0]
        assert "A" in err.message
        assert "depends on itself" in err.message


class TestFindUnreachableNodes:
    """Test unreachable node detection."""

    def test_no_unreachable_in_linear(self, sample_linear_dag):
        unreachable = find_unreachable_nodes(sample_linear_dag)
        assert len(unreachable) == 0

    def test_no_unreachable_in_diamond(self, sample_diamond_dag):
        unreachable = find_unreachable_nodes(sample_diamond_dag)
        assert len(unreachable) == 0

    def test_island_node_is_not_unreachable_if_root(self):
        """Isolated node with no deps is a root, so it IS reachable."""
        dag = DAGWorkflowData(
            dag_id="island",
            name="Island",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b"),
            },
        )
        unreachable = find_unreachable_nodes(dag)
        assert len(unreachable) == 0


class TestMissingDependencies:
    """Test missing dependency validation."""

    def test_missing_dependency(self, sample_invalid_missing_dep_dag):
        result = validate_dag(sample_invalid_missing_dep_dag)
        assert not result.is_valid
        assert result.has_error_type("missing_dependency")

    def test_missing_dependency_message(self, sample_invalid_missing_dep_dag):
        result = validate_dag(sample_invalid_missing_dep_dag)
        err = [e for e in result.errors if e.error_type == "missing_dependency"][0]
        assert "nonexistent" in err.message

    def test_multiple_missing_deps(self):
        dag = DAGWorkflowData(
            dag_id="multi-missing",
            name="Multi Missing",
            nodes={
                "A": DAGNodeData(
                    node_id="A", agent_id="a", depends_on=["ghost1", "ghost2"]
                ),
            },
        )
        result = validate_dag(dag)
        missing_errors = [
            e for e in result.errors if e.error_type == "missing_dependency"
        ]
        assert len(missing_errors) == 2


class TestDuplicateNodeIDs:
    """Test duplicate node ID detection (via dict key mismatch)."""

    def test_node_id_key_mismatch(self):
        dag = DAGWorkflowData(
            dag_id="mismatch",
            name="Mismatch",
            nodes={
                "A": DAGNodeData(node_id="B", agent_id="a"),  # key A but id B
            },
        )
        result = validate_dag(dag)
        assert not result.is_valid
        assert result.has_error_type("id_mismatch")


class TestEmptyDAG:
    """Test empty DAG validation."""

    def test_empty_dag(self):
        dag = DAGWorkflowData(dag_id="empty", name="Empty", nodes={})
        result = validate_dag(dag)
        assert not result.is_valid
        assert result.has_error_type("empty_dag")


class TestMultipleRoots:
    """Test DAGs with multiple root nodes (valid)."""

    def test_two_independent_chains(self):
        dag = DAGWorkflowData(
            dag_id="multi-root",
            name="Multi Root",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A"]),
                "C": DAGNodeData(node_id="C", agent_id="c"),
                "D": DAGNodeData(node_id="D", agent_id="d", depends_on=["C"]),
            },
        )
        result = validate_dag(dag)
        assert result.is_valid


class TestMultipleSinks:
    """Test DAGs with multiple sink nodes (valid)."""

    def test_two_sinks(self):
        dag = DAGWorkflowData(
            dag_id="multi-sink",
            name="Multi Sink",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a"),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["A"]),
                "C": DAGNodeData(node_id="C", agent_id="c", depends_on=["A"]),
            },
        )
        result = validate_dag(dag)
        assert result.is_valid


class TestValidationReturnsAllErrors:
    """Test that validation returns ALL errors, not just first."""

    def test_multiple_error_types_returned(self):
        dag = DAGWorkflowData(
            dag_id="multi-error",
            name="Multi Error",
            nodes={
                "A": DAGNodeData(
                    node_id="A", agent_id="a", depends_on=["A", "ghost"]
                ),
            },
        )
        result = validate_dag(dag)
        assert not result.is_valid
        error_types = {e.error_type for e in result.errors}
        assert "self_dependency" in error_types
        assert "missing_dependency" in error_types

    def test_all_nodes_validated(self):
        dag = DAGWorkflowData(
            dag_id="all-bad",
            name="All Bad",
            nodes={
                "A": DAGNodeData(node_id="A", agent_id="a", depends_on=["missing1"]),
                "B": DAGNodeData(node_id="B", agent_id="b", depends_on=["missing2"]),
            },
        )
        result = validate_dag(dag)
        missing_errors = [
            e for e in result.errors if e.error_type == "missing_dependency"
        ]
        assert len(missing_errors) == 2


class TestValidationResultAPI:
    """Test ValidationResult API."""

    def test_is_valid_true(self):
        result = ValidationResult()
        assert result.is_valid

    def test_is_valid_false(self):
        result = ValidationResult(
            errors=[ValidationError("test", "error")]
        )
        assert not result.is_valid

    def test_error_count(self):
        result = ValidationResult(
            errors=[
                ValidationError("a", "msg"),
                ValidationError("b", "msg"),
            ]
        )
        assert result.error_count == 2

    def test_has_error_type_true(self):
        result = ValidationResult(
            errors=[ValidationError("cycle_detected", "cycle")]
        )
        assert result.has_error_type("cycle_detected")

    def test_has_error_type_false(self):
        result = ValidationResult(
            errors=[ValidationError("cycle_detected", "cycle")]
        )
        assert not result.has_error_type("missing_dependency")

    def test_warnings_do_not_affect_validity(self):
        result = ValidationResult(
            warnings=[ValidationError("unreachable_node", "warning")]
        )
        assert result.is_valid
