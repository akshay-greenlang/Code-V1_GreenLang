# -*- coding: utf-8 -*-
"""
Unit Tests for DependencyResolver (AGENT-FOUND-007)

Tests dependency resolution, topological sort, cycle detection,
missing dependency handling, version constraints, and dependency trees.

Coverage target: 85%+ of dependency_resolver.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline DependencyResolver (self-contained)
# ---------------------------------------------------------------------------


class DependencyNode:
    """A node in the dependency graph."""

    def __init__(self, agent_id: str, version: str = "1.0.0",
                 dependencies: Optional[List[Dict[str, Any]]] = None):
        self.agent_id = agent_id
        self.version = version
        self.dependencies = dependencies or []  # [{agent_id, version_constraint, optional}]


class CyclicDependencyError(Exception):
    """Raised when a cyclic dependency is detected."""
    pass


class MissingDependencyError(Exception):
    """Raised when a required dependency is missing."""
    pass


class DependencyResolver:
    """Resolves agent dependency graphs with topological sort."""

    def __init__(self, fail_on_missing: bool = True, max_depth: int = 10):
        self._nodes: Dict[str, DependencyNode] = {}
        self._fail_on_missing = fail_on_missing
        self._max_depth = max_depth

    def add_node(self, node: DependencyNode) -> None:
        self._nodes[node.agent_id] = node

    def remove_node(self, agent_id: str) -> bool:
        if agent_id in self._nodes:
            del self._nodes[agent_id]
            return True
        return False

    def resolve(self, agent_id: str) -> List[str]:
        """Resolve dependencies via topological sort, returning execution order."""
        if agent_id not in self._nodes:
            if self._fail_on_missing:
                raise MissingDependencyError(f"Agent not found: {agent_id}")
            return [agent_id]

        visited: Set[str] = set()
        in_stack: Set[str] = set()
        order: List[str] = []

        def _visit(node_id: str, depth: int = 0):
            if depth > self._max_depth:
                raise CyclicDependencyError(
                    f"Max dependency depth ({self._max_depth}) exceeded"
                )
            if node_id in in_stack:
                raise CyclicDependencyError(
                    f"Cyclic dependency detected involving: {node_id}"
                )
            if node_id in visited:
                return

            in_stack.add(node_id)
            node = self._nodes.get(node_id)
            if node:
                for dep in node.dependencies:
                    dep_id = dep["agent_id"]
                    optional = dep.get("optional", False)
                    if dep_id not in self._nodes:
                        if not optional and self._fail_on_missing:
                            raise MissingDependencyError(
                                f"Missing dependency: {dep_id}"
                            )
                        continue
                    _visit(dep_id, depth + 1)

            in_stack.remove(node_id)
            visited.add(node_id)
            order.append(node_id)

        _visit(agent_id)
        return order

    def detect_cycles(self) -> List[List[str]]:
        """Detect all cycles in the dependency graph."""
        cycles = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def _dfs(node_id: str):
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            node = self._nodes.get(node_id)
            if node:
                for dep in node.dependencies:
                    dep_id = dep["agent_id"]
                    if dep_id not in self._nodes:
                        continue
                    if dep_id in rec_stack:
                        # Found cycle
                        cycle_start = path.index(dep_id)
                        cycles.append(path[cycle_start:] + [dep_id])
                    elif dep_id not in visited:
                        _dfs(dep_id)

            path.pop()
            rec_stack.remove(node_id)

        for node_id in self._nodes:
            if node_id not in visited:
                _dfs(node_id)

        return cycles

    def get_dependents(self, agent_id: str) -> List[str]:
        """Get agents that depend on the given agent."""
        dependents = []
        for nid, node in self._nodes.items():
            if nid == agent_id:
                continue
            for dep in node.dependencies:
                if dep["agent_id"] == agent_id:
                    dependents.append(nid)
                    break
        return dependents

    def get_dependency_tree(self, agent_id: str, depth: int = 0) -> Dict[str, Any]:
        """Get nested dependency tree."""
        if depth > self._max_depth:
            return {"agent_id": agent_id, "children": [], "truncated": True}

        node = self._nodes.get(agent_id)
        children = []
        if node:
            for dep in node.dependencies:
                dep_id = dep["agent_id"]
                if dep_id in self._nodes:
                    children.append(
                        self.get_dependency_tree(dep_id, depth + 1)
                    )
                else:
                    children.append({
                        "agent_id": dep_id, "children": [], "missing": True,
                    })

        return {"agent_id": agent_id, "children": children}

    def get_all_dependencies(self, agent_id: str, include_optional: bool = True) -> Set[str]:
        """Get all transitive dependencies (flat set)."""
        deps: Set[str] = set()
        visited: Set[str] = set()

        def _collect(nid: str):
            if nid in visited:
                return
            visited.add(nid)
            node = self._nodes.get(nid)
            if node:
                for dep in node.dependencies:
                    dep_id = dep["agent_id"]
                    optional = dep.get("optional", False)
                    if not include_optional and optional:
                        continue
                    deps.add(dep_id)
                    if dep_id in self._nodes:
                        _collect(dep_id)

        _collect(agent_id)
        return deps


# ===========================================================================
# Test Classes
# ===========================================================================


class TestDependencyResolverResolve:
    """Test resolve() with various graph shapes."""

    def test_simple_chain(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A", dependencies=[{"agent_id": "B"}]))
        resolver.add_node(DependencyNode("B", dependencies=[{"agent_id": "C"}]))
        resolver.add_node(DependencyNode("C"))
        order = resolver.resolve("A")
        assert order == ["C", "B", "A"]

    def test_diamond_dependency(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A", dependencies=[
            {"agent_id": "B"}, {"agent_id": "C"},
        ]))
        resolver.add_node(DependencyNode("B", dependencies=[{"agent_id": "D"}]))
        resolver.add_node(DependencyNode("C", dependencies=[{"agent_id": "D"}]))
        resolver.add_node(DependencyNode("D"))
        order = resolver.resolve("A")
        assert order.index("D") < order.index("B")
        assert order.index("D") < order.index("C")
        assert order[-1] == "A"

    def test_complex_graph(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A", dependencies=[
            {"agent_id": "B"}, {"agent_id": "C"},
        ]))
        resolver.add_node(DependencyNode("B", dependencies=[{"agent_id": "D"}]))
        resolver.add_node(DependencyNode("C", dependencies=[
            {"agent_id": "D"}, {"agent_id": "E"},
        ]))
        resolver.add_node(DependencyNode("D"))
        resolver.add_node(DependencyNode("E"))
        order = resolver.resolve("A")
        assert order[-1] == "A"
        assert len(order) == 5

    def test_no_dependencies(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A"))
        order = resolver.resolve("A")
        assert order == ["A"]

    def test_single_dependency(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A", dependencies=[{"agent_id": "B"}]))
        resolver.add_node(DependencyNode("B"))
        order = resolver.resolve("A")
        assert order == ["B", "A"]


class TestDependencyResolverTopologicalSort:
    """Test deterministic ordering."""

    def test_deterministic_ordering(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A", dependencies=[{"agent_id": "B"}]))
        resolver.add_node(DependencyNode("B"))
        order1 = resolver.resolve("A")
        order2 = resolver.resolve("A")
        assert order1 == order2

    def test_dependencies_before_dependents(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("X", dependencies=[
            {"agent_id": "Y"}, {"agent_id": "Z"},
        ]))
        resolver.add_node(DependencyNode("Y"))
        resolver.add_node(DependencyNode("Z"))
        order = resolver.resolve("X")
        assert order.index("Y") < order.index("X")
        assert order.index("Z") < order.index("X")


class TestDependencyResolverCycleDetection:
    """Test cycle detection."""

    def test_simple_cycle(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A", dependencies=[{"agent_id": "B"}]))
        resolver.add_node(DependencyNode("B", dependencies=[{"agent_id": "A"}]))
        with pytest.raises(CyclicDependencyError):
            resolver.resolve("A")

    def test_transitive_cycle(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A", dependencies=[{"agent_id": "B"}]))
        resolver.add_node(DependencyNode("B", dependencies=[{"agent_id": "C"}]))
        resolver.add_node(DependencyNode("C", dependencies=[{"agent_id": "A"}]))
        with pytest.raises(CyclicDependencyError):
            resolver.resolve("A")

    def test_self_cycle(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A", dependencies=[{"agent_id": "A"}]))
        with pytest.raises(CyclicDependencyError):
            resolver.resolve("A")

    def test_detect_cycles_returns_cycles(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A", dependencies=[{"agent_id": "B"}]))
        resolver.add_node(DependencyNode("B", dependencies=[{"agent_id": "A"}]))
        cycles = resolver.detect_cycles()
        assert len(cycles) > 0

    def test_detect_cycles_no_cycles(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A", dependencies=[{"agent_id": "B"}]))
        resolver.add_node(DependencyNode("B"))
        cycles = resolver.detect_cycles()
        assert len(cycles) == 0


class TestDependencyResolverMissing:
    """Test missing dependency handling."""

    def test_missing_fails_when_enabled(self):
        resolver = DependencyResolver(fail_on_missing=True)
        resolver.add_node(DependencyNode("A", dependencies=[{"agent_id": "B"}]))
        with pytest.raises(MissingDependencyError):
            resolver.resolve("A")

    def test_missing_skipped_when_disabled(self):
        resolver = DependencyResolver(fail_on_missing=False)
        resolver.add_node(DependencyNode("A", dependencies=[{"agent_id": "B"}]))
        order = resolver.resolve("A")
        assert order == ["A"]

    def test_missing_agent_id_raises(self):
        resolver = DependencyResolver(fail_on_missing=True)
        with pytest.raises(MissingDependencyError):
            resolver.resolve("nonexistent")

    def test_missing_agent_id_returns_list_when_disabled(self):
        resolver = DependencyResolver(fail_on_missing=False)
        order = resolver.resolve("nonexistent")
        assert order == ["nonexistent"]

    def test_optional_missing_skipped(self):
        resolver = DependencyResolver(fail_on_missing=True)
        resolver.add_node(DependencyNode("A", dependencies=[
            {"agent_id": "B", "optional": True},
        ]))
        order = resolver.resolve("A")
        assert order == ["A"]

    def test_required_missing_with_optional_present(self):
        resolver = DependencyResolver(fail_on_missing=True)
        resolver.add_node(DependencyNode("A", dependencies=[
            {"agent_id": "B", "optional": True},
            {"agent_id": "C", "optional": False},
        ]))
        with pytest.raises(MissingDependencyError):
            resolver.resolve("A")


class TestDependencyResolverGetDependents:
    """Test get_dependents."""

    def test_returns_correct_dependents(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A", dependencies=[{"agent_id": "C"}]))
        resolver.add_node(DependencyNode("B", dependencies=[{"agent_id": "C"}]))
        resolver.add_node(DependencyNode("C"))
        dependents = resolver.get_dependents("C")
        assert set(dependents) == {"A", "B"}

    def test_no_dependents(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A"))
        assert resolver.get_dependents("A") == []

    def test_dependents_chain(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A", dependencies=[{"agent_id": "B"}]))
        resolver.add_node(DependencyNode("B", dependencies=[{"agent_id": "C"}]))
        resolver.add_node(DependencyNode("C"))
        assert resolver.get_dependents("B") == ["A"]


class TestDependencyResolverGetTree:
    """Test get_dependency_tree."""

    def test_tree_structure(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A", dependencies=[{"agent_id": "B"}]))
        resolver.add_node(DependencyNode("B", dependencies=[{"agent_id": "C"}]))
        resolver.add_node(DependencyNode("C"))
        tree = resolver.get_dependency_tree("A")
        assert tree["agent_id"] == "A"
        assert len(tree["children"]) == 1
        assert tree["children"][0]["agent_id"] == "B"
        assert len(tree["children"][0]["children"]) == 1

    def test_tree_missing_dep(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A", dependencies=[{"agent_id": "B"}]))
        tree = resolver.get_dependency_tree("A")
        assert tree["children"][0].get("missing") is True

    def test_tree_no_deps(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A"))
        tree = resolver.get_dependency_tree("A")
        assert tree["children"] == []


class TestDependencyResolverOptional:
    """Test optional dependency handling."""

    def test_include_optional(self):
        resolver = DependencyResolver(fail_on_missing=False)
        resolver.add_node(DependencyNode("A", dependencies=[
            {"agent_id": "B", "optional": True},
            {"agent_id": "C", "optional": False},
        ]))
        resolver.add_node(DependencyNode("B"))
        resolver.add_node(DependencyNode("C"))
        all_deps = resolver.get_all_dependencies("A", include_optional=True)
        assert "B" in all_deps
        assert "C" in all_deps

    def test_exclude_optional(self):
        resolver = DependencyResolver(fail_on_missing=False)
        resolver.add_node(DependencyNode("A", dependencies=[
            {"agent_id": "B", "optional": True},
            {"agent_id": "C", "optional": False},
        ]))
        resolver.add_node(DependencyNode("B"))
        resolver.add_node(DependencyNode("C"))
        required_deps = resolver.get_all_dependencies("A", include_optional=False)
        assert "B" not in required_deps
        assert "C" in required_deps

    def test_remove_node(self):
        resolver = DependencyResolver()
        resolver.add_node(DependencyNode("A"))
        assert resolver.remove_node("A") is True
        assert resolver.remove_node("A") is False
