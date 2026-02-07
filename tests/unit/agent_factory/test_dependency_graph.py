# -*- coding: utf-8 -*-
"""
Unit tests for Agent Factory Dependency Graph: directed graph construction,
dependency queries, topological sorting, parallel group detection,
cycle detection, and graph visualization.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import pytest


# ============================================================================
# Inline Implementations (contract definitions)
# ============================================================================


class CycleDetectedError(Exception):
    def __init__(self, cycle: List[str]) -> None:
        self.cycle = cycle
        super().__init__(f"Cycle detected: {' -> '.join(cycle)}")


class DependencyGraph:
    """Directed acyclic graph of agent dependencies."""

    def __init__(self) -> None:
        self._nodes: Set[str] = set()
        self._edges: Dict[str, Set[str]] = defaultdict(set)
        self._reverse: Dict[str, Set[str]] = defaultdict(set)
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def add_agent(self, key: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._nodes.add(key)
        if metadata:
            self._metadata[key] = metadata

    def add_dependency(self, dependent: str, dependency: str) -> None:
        if dependent not in self._nodes:
            self.add_agent(dependent)
        if dependency not in self._nodes:
            self.add_agent(dependency)
        self._edges[dependent].add(dependency)
        self._reverse[dependency].add(dependent)

    def get_dependencies(self, key: str) -> Set[str]:
        return set(self._edges.get(key, set()))

    def get_dependents(self, key: str) -> Set[str]:
        return set(self._reverse.get(key, set()))

    def get_transitive_dependencies(self, key: str) -> Set[str]:
        result: Set[str] = set()
        queue = deque(self._edges.get(key, set()))
        while queue:
            dep = queue.popleft()
            if dep not in result:
                result.add(dep)
                queue.extend(self._edges.get(dep, set()))
        return result

    def topological_sort(self) -> List[str]:
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        for node, deps in self._edges.items():
            for dep in deps:
                in_degree[dep] = in_degree.get(dep, 0)
            # increment for dependents
        for node, deps in self._edges.items():
            for dep in deps:
                pass  # edges go from dependent -> dependency

        # Kahn's algorithm on reversed edges (dependency -> dependent order)
        # We want dependencies first, so treat _edges as "needs"
        in_deg: Dict[str, int] = {n: 0 for n in self._nodes}
        for node in self._nodes:
            for dep in self._edges.get(node, set()):
                in_deg[node] = in_deg.get(node, 0)
                # node depends on dep, so node can't start until dep completes
                pass

        # Recalculate: in_degree counts how many dependencies each node has
        in_deg = {n: len(self._edges.get(n, set())) for n in self._nodes}
        queue = deque([n for n in self._nodes if in_deg[n] == 0])
        result: List[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for dependent in self._reverse.get(node, set()):
                in_deg[dependent] -= 1
                if in_deg[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._nodes):
            raise CycleDetectedError(self._find_cycle())

        return result

    def parallel_groups(self) -> List[Set[str]]:
        in_deg = {n: len(self._edges.get(n, set())) for n in self._nodes}
        current = {n for n in self._nodes if in_deg[n] == 0}
        groups: List[Set[str]] = []

        while current:
            groups.append(set(current))
            next_level: Set[str] = set()
            for node in current:
                for dependent in self._reverse.get(node, set()):
                    in_deg[dependent] -= 1
                    if in_deg[dependent] == 0:
                        next_level.add(dependent)
            current = next_level

        total = sum(len(g) for g in groups)
        if total != len(self._nodes):
            raise CycleDetectedError(self._find_cycle())

        return groups

    def detect_cycle(self) -> Optional[List[str]]:
        try:
            self.topological_sort()
            return None
        except CycleDetectedError as exc:
            return exc.cycle

    def find_all_cycles(self) -> List[List[str]]:
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            stack.add(node)
            path.append(node)
            for dep in self._edges.get(node, set()):
                if dep not in visited:
                    dfs(dep)
                elif dep in stack:
                    idx = path.index(dep)
                    cycles.append(path[idx:] + [dep])
            path.pop()
            stack.discard(node)

        for n in self._nodes:
            if n not in visited:
                dfs(n)
        return cycles

    def _find_cycle(self) -> List[str]:
        visited: Set[str] = set()
        stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> Optional[List[str]]:
            visited.add(node)
            stack.add(node)
            path.append(node)
            for dep in self._edges.get(node, set()):
                if dep not in visited:
                    result = dfs(dep)
                    if result:
                        return result
                elif dep in stack:
                    idx = path.index(dep)
                    return path[idx:] + [dep]
            path.pop()
            stack.discard(node)
            return None

        for n in self._nodes:
            if n not in visited:
                cycle = dfs(n)
                if cycle:
                    return cycle
        return []

    def to_dot(self) -> str:
        lines = ["digraph agents {"]
        for node in sorted(self._nodes):
            lines.append(f'  "{node}";')
        for node, deps in sorted(self._edges.items()):
            for dep in sorted(deps):
                lines.append(f'  "{node}" -> "{dep}";')
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        lines = ["graph TD"]
        for node, deps in sorted(self._edges.items()):
            for dep in sorted(deps):
                lines.append(f"  {node} --> {dep}")
        return "\n".join(lines)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(deps) for deps in self._edges.values())


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def graph() -> DependencyGraph:
    return DependencyGraph()


@pytest.fixture
def linear_graph() -> DependencyGraph:
    """A -> B -> C (A depends on B, B depends on C)."""
    g = DependencyGraph()
    g.add_agent("A")
    g.add_agent("B")
    g.add_agent("C")
    g.add_dependency("A", "B")
    g.add_dependency("B", "C")
    return g


@pytest.fixture
def diamond_graph() -> DependencyGraph:
    """
    Top depends on Left and Right.
    Left and Right both depend on Bottom.
    """
    g = DependencyGraph()
    for n in ["top", "left", "right", "bottom"]:
        g.add_agent(n)
    g.add_dependency("top", "left")
    g.add_dependency("top", "right")
    g.add_dependency("left", "bottom")
    g.add_dependency("right", "bottom")
    return g


# ============================================================================
# Tests
# ============================================================================


class TestDependencyGraph:
    """Tests for the dependency graph core operations."""

    def test_graph_add_agent(self, graph: DependencyGraph) -> None:
        """Adding agents increases node count."""
        graph.add_agent("agent-a")
        graph.add_agent("agent-b")
        assert graph.node_count == 2

    def test_graph_add_agent_with_metadata(
        self, graph: DependencyGraph
    ) -> None:
        """Agent metadata is stored."""
        graph.add_agent("agent-a", metadata={"version": "1.0.0"})
        assert graph._metadata["agent-a"]["version"] == "1.0.0"

    def test_graph_add_dependency(self, graph: DependencyGraph) -> None:
        """Adding a dependency creates an edge."""
        graph.add_agent("A")
        graph.add_agent("B")
        graph.add_dependency("A", "B")
        assert graph.edge_count == 1

    def test_graph_add_dependency_auto_creates_nodes(
        self, graph: DependencyGraph
    ) -> None:
        """Adding a dependency for unregistered nodes auto-creates them."""
        graph.add_dependency("X", "Y")
        assert graph.node_count == 2

    def test_graph_get_dependencies(
        self, linear_graph: DependencyGraph
    ) -> None:
        """get_dependencies returns direct dependencies."""
        deps = linear_graph.get_dependencies("A")
        assert deps == {"B"}

    def test_graph_get_dependents(
        self, linear_graph: DependencyGraph
    ) -> None:
        """get_dependents returns direct dependents."""
        dependents = linear_graph.get_dependents("B")
        assert dependents == {"A"}

    def test_graph_transitive_dependencies(
        self, linear_graph: DependencyGraph
    ) -> None:
        """Transitive dependencies traverse the full chain."""
        trans = linear_graph.get_transitive_dependencies("A")
        assert trans == {"B", "C"}

    def test_graph_transitive_no_deps(
        self, linear_graph: DependencyGraph
    ) -> None:
        """Leaf nodes have no transitive dependencies."""
        trans = linear_graph.get_transitive_dependencies("C")
        assert trans == set()

    def test_topological_sort_simple(
        self, linear_graph: DependencyGraph
    ) -> None:
        """Topological sort respects dependency order."""
        order = linear_graph.topological_sort()
        assert order.index("C") < order.index("B")
        assert order.index("B") < order.index("A")

    def test_topological_sort_complex(
        self, diamond_graph: DependencyGraph
    ) -> None:
        """Diamond graph is sorted with bottom before left/right before top."""
        order = diamond_graph.topological_sort()
        assert order.index("bottom") < order.index("left")
        assert order.index("bottom") < order.index("right")
        assert order.index("left") < order.index("top")
        assert order.index("right") < order.index("top")

    def test_parallel_group_detection(
        self, diamond_graph: DependencyGraph
    ) -> None:
        """Parallel groups identify nodes that can execute concurrently."""
        groups = diamond_graph.parallel_groups()
        assert len(groups) == 3
        assert groups[0] == {"bottom"}
        assert groups[1] == {"left", "right"}
        assert groups[2] == {"top"}

    def test_cycle_detector_no_cycle(
        self, linear_graph: DependencyGraph
    ) -> None:
        """detect_cycle returns None for an acyclic graph."""
        assert linear_graph.detect_cycle() is None

    def test_cycle_detector_simple_cycle(self) -> None:
        """Simple A -> B -> A cycle is detected."""
        g = DependencyGraph()
        g.add_dependency("A", "B")
        g.add_dependency("B", "A")
        cycle = g.detect_cycle()
        assert cycle is not None
        assert len(cycle) >= 2

    def test_cycle_detector_complex_cycle(self) -> None:
        """Three-node cycle A -> B -> C -> A is detected."""
        g = DependencyGraph()
        g.add_dependency("A", "B")
        g.add_dependency("B", "C")
        g.add_dependency("C", "A")
        cycle = g.detect_cycle()
        assert cycle is not None

    def test_cycle_detector_diagnostic_path(self) -> None:
        """Cycle detection returns the cycle path."""
        g = DependencyGraph()
        g.add_dependency("X", "Y")
        g.add_dependency("Y", "Z")
        g.add_dependency("Z", "X")
        cycle = g.detect_cycle()
        assert cycle is not None
        # The cycle should form a loop back to the starting node
        assert cycle[0] == cycle[-1]

    def test_cycle_detector_all_cycles(self) -> None:
        """find_all_cycles returns all distinct cycles."""
        g = DependencyGraph()
        g.add_dependency("A", "B")
        g.add_dependency("B", "A")
        g.add_dependency("C", "D")
        g.add_dependency("D", "C")
        cycles = g.find_all_cycles()
        assert len(cycles) >= 2

    def test_topological_sort_raises_on_cycle(self) -> None:
        """Topological sort raises CycleDetectedError on cycles."""
        g = DependencyGraph()
        g.add_dependency("A", "B")
        g.add_dependency("B", "A")
        with pytest.raises(CycleDetectedError):
            g.topological_sort()


class TestGraphVisualization:
    """Tests for graph visualization output."""

    def test_visualizer_dot_output(
        self, linear_graph: DependencyGraph
    ) -> None:
        """DOT output contains nodes and edges."""
        dot = linear_graph.to_dot()
        assert "digraph agents {" in dot
        assert '"A"' in dot
        assert '"A" -> "B"' in dot

    def test_visualizer_mermaid_output(
        self, linear_graph: DependencyGraph
    ) -> None:
        """Mermaid output contains graph definition and edges."""
        md = linear_graph.to_mermaid()
        assert "graph TD" in md
        assert "A --> B" in md
        assert "B --> C" in md
