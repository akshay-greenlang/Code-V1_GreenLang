# -*- coding: utf-8 -*-
"""
Cycle Detector - Detect circular dependencies in the agent dependency graph.

Uses depth-first search with three-color marking (WHITE, GRAY, BLACK) to
detect all cycles in the graph. Returns diagnostic paths showing the full
cycle and suggests which edges to remove to break cycles.

Example:
    >>> detector = CycleDetector()
    >>> result = detector.detect_all_cycles(graph)
    >>> if result.has_cycle:
    ...     for cycle in result.cycles:
    ...         print(" -> ".join(cycle))

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from greenlang.infrastructure.agent_factory.dependencies.graph import DependencyGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color enum for DFS
# ---------------------------------------------------------------------------


class _Color(Enum):
    """DFS vertex coloring for cycle detection."""

    WHITE = "white"
    """Not yet visited."""

    GRAY = "gray"
    """Currently being processed (on the recursion stack)."""

    BLACK = "black"
    """Fully processed (all descendants explored)."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class CycleDetectionResult:
    """Outcome of cycle detection on a dependency graph.

    Attributes:
        has_cycle: True if at least one cycle was found.
        cycles: List of cycle paths (each path is a list of agent keys).
        affected_agents: Set of all agents involved in any cycle.
    """

    has_cycle: bool = False
    cycles: List[List[str]] = field(default_factory=list)
    affected_agents: Set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class CycleDetector:
    """Detect circular dependencies in a DependencyGraph.

    Uses DFS with three-color marking to find all cycles. Each detected
    cycle is reported as a path of agent keys.
    """

    def detect_all_cycles(self, graph: DependencyGraph) -> CycleDetectionResult:
        """Detect all cycles in the dependency graph.

        Performs a full DFS from every unvisited node to ensure all
        components are covered.

        Args:
            graph: The dependency graph to check.

        Returns:
            CycleDetectionResult with all found cycles.
        """
        all_agents = graph.all_agents()
        color: Dict[str, _Color] = {key: _Color.WHITE for key in all_agents}
        parent: Dict[str, Optional[str]] = {key: None for key in all_agents}
        cycles: List[List[str]] = []

        for agent in all_agents:
            if color[agent] == _Color.WHITE:
                self._dfs(graph, agent, color, parent, [], cycles)

        # Deduplicate cycles (same cycle may be found from different start nodes)
        unique_cycles = self._deduplicate_cycles(cycles)
        affected: Set[str] = set()
        for cycle in unique_cycles:
            affected.update(cycle)

        result = CycleDetectionResult(
            has_cycle=len(unique_cycles) > 0,
            cycles=unique_cycles,
            affected_agents=affected,
        )

        if result.has_cycle:
            logger.warning(
                "Detected %d cycle(s) involving %d agents",
                len(unique_cycles),
                len(affected),
            )
        else:
            logger.debug("No cycles detected in dependency graph")

        return result

    def suggest_resolution(
        self, graph: DependencyGraph, result: CycleDetectionResult
    ) -> List[Dict[str, str]]:
        """Suggest which edges to remove to break all cycles.

        For each cycle, suggests removing the edge that would have the
        least impact (fewest dependents on the target node).

        Args:
            graph: The dependency graph.
            result: Cycle detection result.

        Returns:
            List of dicts with from_key, to_key, and reason.
        """
        suggestions: List[Dict[str, str]] = []
        seen_edges: Set[Tuple[str, str]] = set()

        for cycle in result.cycles:
            best_edge = self._find_best_edge_to_remove(graph, cycle)
            if best_edge and best_edge not in seen_edges:
                seen_edges.add(best_edge)
                suggestions.append({
                    "from_key": best_edge[0],
                    "to_key": best_edge[1],
                    "reason": (
                        f"Removing {best_edge[0]} -> {best_edge[1]} "
                        f"breaks cycle: {' -> '.join(cycle)}"
                    ),
                })

        return suggestions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dfs(
        self,
        graph: DependencyGraph,
        node: str,
        color: Dict[str, _Color],
        parent: Dict[str, Optional[str]],
        path: List[str],
        cycles: List[List[str]],
    ) -> None:
        """DFS traversal with cycle detection.

        When a GRAY node is encountered, a cycle is found. The cycle
        path is extracted from the current path.
        """
        color[node] = _Color.GRAY
        path.append(node)

        for neighbor in graph.get_dependencies(node):
            if color.get(neighbor) == _Color.GRAY:
                # Found a cycle: extract the cycle path
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)
            elif color.get(neighbor) == _Color.WHITE:
                parent[neighbor] = node
                self._dfs(graph, neighbor, color, parent, path, cycles)

        path.pop()
        color[node] = _Color.BLACK

    def _deduplicate_cycles(self, cycles: List[List[str]]) -> List[List[str]]:
        """Remove duplicate cycle paths.

        Two cycles are considered the same if they contain the same set
        of edges (regardless of starting node).
        """
        unique: List[List[str]] = []
        seen: Set[frozenset[Tuple[str, str]]] = set()

        for cycle in cycles:
            if len(cycle) < 2:
                continue
            edges: Set[Tuple[str, str]] = set()
            for i in range(len(cycle) - 1):
                edges.add((cycle[i], cycle[i + 1]))
            edge_key = frozenset(edges)
            if edge_key not in seen:
                seen.add(edge_key)
                unique.append(cycle)

        return unique

    def _find_best_edge_to_remove(
        self, graph: DependencyGraph, cycle: List[str]
    ) -> Optional[Tuple[str, str]]:
        """Find the edge in a cycle whose removal has least impact.

        Prefers removing edges where the target node has the fewest
        dependents overall.
        """
        if len(cycle) < 2:
            return None

        best_edge: Optional[Tuple[str, str]] = None
        min_impact = float("inf")

        for i in range(len(cycle) - 1):
            from_key = cycle[i]
            to_key = cycle[i + 1]
            # Impact = number of dependents of the target node
            dependents = graph.get_dependents(to_key)
            impact = len(dependents)
            if impact < min_impact:
                min_impact = impact
                best_edge = (from_key, to_key)

        return best_edge
