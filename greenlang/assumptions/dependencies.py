# -*- coding: utf-8 -*-
"""
Dependency Tracker - AGENT-FOUND-004: Assumptions Registry

Tracks dependencies between assumptions and between assumptions and
calculations. Provides upstream/downstream traversal, cycle detection,
and impact analysis for change propagation.

Zero-Hallucination Guarantees:
    - All graph operations are deterministic
    - Cycle detection uses standard DFS algorithm
    - Impact analysis is purely structural (no predictions)

Example:
    >>> from greenlang.assumptions.dependencies import DependencyTracker
    >>> tracker = DependencyTracker()
    >>> tracker.register_dependency("ef.elec", "grid.mix")
    >>> tracker.register_calculation("scope2_calc", ["ef.elec", "activity.kwh"])
    >>> upstream = tracker.get_upstream("ef.elec")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-004 Assumptions Registry
Status: Production Ready
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from greenlang.assumptions.models import DependencyNode
from greenlang.assumptions.metrics import record_dependency_depth

logger = logging.getLogger(__name__)


class DependencyTracker:
    """Tracks dependency relationships between assumptions and calculations.

    Maintains a directed graph of assumption-to-assumption and
    assumption-to-calculation dependencies. Supports traversal,
    cycle detection, and impact analysis.

    Attributes:
        _nodes: Mapping of assumption_id to DependencyNode.
        _calculations: Mapping of calculation_id to set of assumption_ids.

    Example:
        >>> tracker = DependencyTracker()
        >>> tracker.register_dependency("ef.nat_gas", "gwp.ch4")
        >>> tracker.register_calculation("emissions_calc", ["ef.nat_gas"])
        >>> print(tracker.get_downstream("gwp.ch4"))
    """

    def __init__(self) -> None:
        """Initialize DependencyTracker."""
        self._nodes: Dict[str, DependencyNode] = {}
        self._calculations: Dict[str, Set[str]] = defaultdict(set)
        logger.info("DependencyTracker initialized")

    def register_dependency(
        self,
        assumption_id: str,
        depends_on: str,
    ) -> None:
        """Register that one assumption depends on another.

        Args:
            assumption_id: The assumption that has the dependency.
            depends_on: The assumption it depends on (upstream).
        """
        # Ensure nodes exist
        self._ensure_node(assumption_id)
        self._ensure_node(depends_on)

        # Add upstream link
        node = self._nodes[assumption_id]
        if depends_on not in node.upstream:
            node.upstream.append(depends_on)

        # Add downstream link
        upstream_node = self._nodes[depends_on]
        if assumption_id not in upstream_node.downstream:
            upstream_node.downstream.append(assumption_id)

        logger.debug(
            "Registered dependency: %s -> %s", assumption_id, depends_on,
        )

    def register_calculation(
        self,
        calculation_id: str,
        assumption_ids: List[str],
    ) -> None:
        """Register that a calculation depends on specific assumptions.

        Args:
            calculation_id: Identifier for the calculation.
            assumption_ids: List of assumption IDs the calculation uses.
        """
        self._calculations[calculation_id] = set(assumption_ids)

        for assumption_id in assumption_ids:
            self._ensure_node(assumption_id)
            node = self._nodes[assumption_id]
            if calculation_id not in node.calculation_ids:
                node.calculation_ids.append(calculation_id)

        logger.debug(
            "Registered calculation %s with %d assumptions",
            calculation_id, len(assumption_ids),
        )

    def get_upstream(self, assumption_id: str) -> List[str]:
        """Get all upstream assumptions (direct dependencies).

        Args:
            assumption_id: The assumption to query.

        Returns:
            List of upstream assumption IDs.
        """
        node = self._nodes.get(assumption_id)
        if node is None:
            return []
        return list(node.upstream)

    def get_downstream(self, assumption_id: str) -> List[str]:
        """Get all downstream assumptions (dependents).

        Args:
            assumption_id: The assumption to query.

        Returns:
            List of downstream assumption IDs.
        """
        node = self._nodes.get(assumption_id)
        if node is None:
            return []
        return list(node.downstream)

    def get_impact(self, assumption_id: str) -> Dict[str, Any]:
        """Analyze the impact of changing an assumption.

        Traverses the dependency graph to find all directly and
        transitively affected assumptions and calculations.

        Args:
            assumption_id: The assumption being changed.

        Returns:
            Dictionary with affected assumptions and calculations.
        """
        affected_assumptions: Set[str] = set()
        affected_calculations: Set[str] = set()

        # BFS through downstream assumptions
        queue = [assumption_id]
        visited: Set[str] = set()
        depth = 0

        while queue:
            next_queue: List[str] = []
            for aid in queue:
                if aid in visited:
                    continue
                visited.add(aid)

                node = self._nodes.get(aid)
                if node is None:
                    continue

                # Add this node's calculations
                affected_calculations.update(node.calculation_ids)

                # Add downstream assumptions
                for downstream_id in node.downstream:
                    if downstream_id not in visited:
                        affected_assumptions.add(downstream_id)
                        next_queue.append(downstream_id)

            queue = next_queue
            depth += 1

        # Record depth metric
        record_dependency_depth(depth)

        return {
            "assumption_id": assumption_id,
            "affected_assumptions": sorted(affected_assumptions),
            "affected_calculations": sorted(affected_calculations),
            "total_affected": len(affected_assumptions) + len(affected_calculations),
            "max_depth": depth,
        }

    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the dependency graph using DFS.

        Returns:
            List of cycles found. Each cycle is a list of assumption IDs.
        """
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def _dfs(node_id: str) -> None:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            node = self._nodes.get(node_id)
            if node:
                for upstream_id in node.upstream:
                    if upstream_id not in visited:
                        _dfs(upstream_id)
                    elif upstream_id in rec_stack:
                        # Found a cycle
                        cycle_start = path.index(upstream_id)
                        cycle = path[cycle_start:] + [upstream_id]
                        cycles.append(cycle)

            path.pop()
            rec_stack.discard(node_id)

        for node_id in self._nodes:
            if node_id not in visited:
                _dfs(node_id)

        if cycles:
            logger.warning("Detected %d dependency cycles", len(cycles))

        return cycles

    def get_calculation_assumptions(
        self,
        calculation_id: str,
        scenario_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get all assumptions needed for a calculation.

        Args:
            calculation_id: The calculation identifier.
            scenario_id: Optional scenario ID for context (metadata only).

        Returns:
            Dictionary of assumption_id -> metadata dict.
        """
        assumption_ids = self._calculations.get(calculation_id, set())
        result: Dict[str, Any] = {}

        for aid in sorted(assumption_ids):
            node = self._nodes.get(aid)
            result[aid] = {
                "assumption_id": aid,
                "has_upstream": bool(node and node.upstream),
                "upstream_count": len(node.upstream) if node else 0,
                "scenario_id": scenario_id,
            }

        return result

    def get_node(self, assumption_id: str) -> Optional[DependencyNode]:
        """Get the dependency node for an assumption.

        Args:
            assumption_id: The assumption identifier.

        Returns:
            DependencyNode or None if not found.
        """
        return self._nodes.get(assumption_id)

    def get_all_nodes(self) -> Dict[str, DependencyNode]:
        """Get all dependency nodes.

        Returns:
            Dictionary of assumption_id to DependencyNode.
        """
        return dict(self._nodes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_node(self, assumption_id: str) -> None:
        """Ensure a node exists in the graph for the given assumption.

        Args:
            assumption_id: The assumption identifier.
        """
        if assumption_id not in self._nodes:
            self._nodes[assumption_id] = DependencyNode(
                assumption_id=assumption_id,
            )


__all__ = [
    "DependencyTracker",
]
