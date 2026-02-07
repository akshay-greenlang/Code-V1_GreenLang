# -*- coding: utf-8 -*-
"""
Topological Resolver - Topological sort and parallel execution planning.

Uses Kahn's algorithm to produce a topological ordering of the dependency
graph, then groups agents by depth level so that agents at the same depth
can execute concurrently.

Example:
    >>> resolver = TopologicalResolver()
    >>> plan = resolver.resolve(graph)
    >>> for group in plan.groups:
    ...     print(f"Parallel group: {group.agent_keys}")

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from greenlang.infrastructure.agent_factory.dependencies.graph import (
    DependencyGraph,
    EdgeType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionGroup:
    """A group of agents that can execute concurrently.

    All agents in a group have their dependencies satisfied by agents
    in earlier groups.

    Attributes:
        depth: Depth level in the dependency graph (0 = no dependencies).
        agent_keys: List of agent keys that can run in parallel.
    """

    depth: int
    agent_keys: List[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """Ordered execution plan produced by topological sort.

    Attributes:
        groups: Ordered list of execution groups (by depth).
        total_agents: Total number of agents in the plan.
        max_parallelism: Largest group size (peak parallel execution).
        topological_order: Flat ordered list of all agent keys.
        warnings: Non-fatal issues detected during resolution.
    """

    groups: List[ExecutionGroup] = field(default_factory=list)
    total_agents: int = 0
    max_parallelism: int = 0
    topological_order: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class TopologicalResolver:
    """Resolve dependency graph into topologically sorted execution plan.

    Uses Kahn's algorithm for BFS-based topological sorting. Supports
    optional dependency handling (skip if not available).

    Attributes:
        include_optional: Whether to include optional dependencies.
        include_test: Whether to include test dependencies.
    """

    def __init__(
        self,
        include_optional: bool = False,
        include_test: bool = False,
    ) -> None:
        """Initialize the resolver.

        Args:
            include_optional: Include optional dependencies in ordering.
            include_test: Include test dependencies in ordering.
        """
        self.include_optional = include_optional
        self.include_test = include_test

    def resolve(self, graph: DependencyGraph) -> ExecutionPlan:
        """Resolve the dependency graph into an execution plan.

        Args:
            graph: The dependency graph to resolve.

        Returns:
            ExecutionPlan with ordered groups for parallel execution.

        Raises:
            ValueError: If the graph contains cycles (should be checked
                with CycleDetector first).
        """
        warnings: List[str] = []
        all_agents = set(graph.all_agents())

        if not all_agents:
            return ExecutionPlan()

        # Validate all dependencies exist
        self._validate_dependencies(graph, all_agents, warnings)

        # Build filtered in-degree map and adjacency
        in_degree: Dict[str, int] = {key: 0 for key in all_agents}
        dependents: Dict[str, List[str]] = {key: [] for key in all_agents}

        for agent_key in all_agents:
            for edge in graph.get_dependency_edges(agent_key):
                if not self._should_include_edge(edge.edge_type):
                    continue
                if edge.to_key not in all_agents:
                    if edge.edge_type == EdgeType.OPTIONAL:
                        warnings.append(
                            f"Optional dependency '{edge.to_key}' for "
                            f"'{agent_key}' not available, skipping."
                        )
                        continue
                    continue
                in_degree[agent_key] += 1
                dependents[edge.to_key].append(agent_key)

        # Kahn's algorithm with depth tracking
        queue: deque[tuple[str, int]] = deque()
        for key in all_agents:
            if in_degree[key] == 0:
                queue.append((key, 0))

        depth_groups: Dict[int, List[str]] = {}
        topological_order: List[str] = []
        processed = 0

        while queue:
            current, depth = queue.popleft()
            topological_order.append(current)
            depth_groups.setdefault(depth, []).append(current)
            processed += 1

            for dependent in dependents.get(current, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append((dependent, depth + 1))

        # Check for cycles (should not happen if CycleDetector was used)
        if processed < len(all_agents):
            unresolved = [k for k in all_agents if in_degree[k] > 0]
            raise ValueError(
                f"Dependency graph contains cycles. Unresolved agents: "
                f"{sorted(unresolved)}"
            )

        # Build execution groups
        groups: List[ExecutionGroup] = []
        for depth in sorted(depth_groups.keys()):
            agents = sorted(depth_groups[depth])
            groups.append(ExecutionGroup(depth=depth, agent_keys=agents))

        max_parallelism = max(len(g.agent_keys) for g in groups) if groups else 0

        logger.info(
            "Resolved execution plan: %d agents, %d groups, max parallelism=%d",
            processed,
            len(groups),
            max_parallelism,
        )

        return ExecutionPlan(
            groups=groups,
            total_agents=processed,
            max_parallelism=max_parallelism,
            topological_order=topological_order,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_include_edge(self, edge_type: EdgeType) -> bool:
        """Determine if an edge type should be included in resolution."""
        if edge_type == EdgeType.RUNTIME:
            return True
        if edge_type == EdgeType.BUILD:
            return True
        if edge_type == EdgeType.OPTIONAL:
            return self.include_optional
        if edge_type == EdgeType.TEST:
            return self.include_test
        return True

    def _validate_dependencies(
        self,
        graph: DependencyGraph,
        all_agents: Set[str],
        warnings: List[str],
    ) -> None:
        """Validate that all required dependencies exist in the graph."""
        for agent_key in all_agents:
            for edge in graph.get_dependency_edges(agent_key):
                if edge.to_key not in all_agents:
                    if edge.edge_type == EdgeType.OPTIONAL:
                        warnings.append(
                            f"Optional dependency '{edge.to_key}' for "
                            f"'{agent_key}' is not in the graph."
                        )
                    else:
                        warnings.append(
                            f"Required dependency '{edge.to_key}' for "
                            f"'{agent_key}' is not in the graph."
                        )
