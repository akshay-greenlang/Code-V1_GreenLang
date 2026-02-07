# -*- coding: utf-8 -*-
"""
Dependency Graph - Directed acyclic graph for agent dependencies.

Implements a DAG using adjacency lists where nodes represent agents and
edges represent dependency relationships with version constraints and
type classification (runtime, build, test, optional).

Example:
    >>> graph = DependencyGraph()
    >>> graph.add_agent("intake", {"version": "1.0.0", "type": "deterministic"})
    >>> graph.add_agent("calc", {"version": "2.0.0", "type": "deterministic"})
    >>> graph.add_dependency("calc", "intake", "^1.0.0")
    >>> deps = graph.get_dependencies("calc")
    >>> assert "intake" in deps

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EdgeType(str, Enum):
    """Classification of a dependency edge."""

    RUNTIME = "runtime"
    """Required at runtime for the agent to function."""

    BUILD = "build"
    """Required only during package build."""

    TEST = "test"
    """Required only for running tests."""

    OPTIONAL = "optional"
    """Enhances functionality but not strictly required."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DependencyEdge:
    """An edge in the dependency graph.

    Attributes:
        from_key: Agent that depends on another.
        to_key: Agent that is depended upon.
        version_constraint: Semver range constraint for the dependency.
        edge_type: Classification of the dependency relationship.
    """

    from_key: str
    to_key: str
    version_constraint: str = "*"
    edge_type: EdgeType = EdgeType.RUNTIME


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class DependencyGraph:
    """Directed acyclic graph of agent dependencies.

    Uses adjacency lists for efficient traversal. Nodes are agent keys
    with associated metadata; edges represent dependency relationships.

    Attributes:
        nodes: Agent keys to metadata mappings.
        edges: Adjacency list (from_key -> list of DependencyEdge).
        reverse_edges: Reverse adjacency list (to_key -> list of from_keys).
    """

    def __init__(self) -> None:
        """Initialize an empty dependency graph."""
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: Dict[str, List[DependencyEdge]] = {}
        self._reverse: Dict[str, List[DependencyEdge]] = {}

    @property
    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Return the total number of edges."""
        return sum(len(edges) for edges in self._edges.values())

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_agent(self, key: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an agent node to the graph.

        Args:
            key: Unique agent key.
            metadata: Optional metadata (version, type, status, etc.).
        """
        if key in self._nodes:
            logger.debug("Updating existing agent node: %s", key)
        self._nodes[key] = metadata or {}
        self._edges.setdefault(key, [])
        self._reverse.setdefault(key, [])
        logger.debug("Added agent node: %s", key)

    def remove_agent(self, key: str) -> None:
        """Remove an agent node and all its edges.

        Args:
            key: Agent key to remove.

        Raises:
            KeyError: If the agent does not exist.
        """
        if key not in self._nodes:
            raise KeyError(f"Agent '{key}' not found in graph")

        # Remove outgoing edges
        for edge in list(self._edges.get(key, [])):
            self._reverse.get(edge.to_key, []).remove(edge)
        self._edges.pop(key, None)

        # Remove incoming edges
        for edge in list(self._reverse.get(key, [])):
            self._edges.get(edge.from_key, []).remove(edge)
        self._reverse.pop(key, None)

        del self._nodes[key]
        logger.debug("Removed agent node: %s", key)

    def has_agent(self, key: str) -> bool:
        """Check if an agent exists in the graph."""
        return key in self._nodes

    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Return metadata for an agent.

        Args:
            key: Agent key.

        Returns:
            Metadata dictionary.

        Raises:
            KeyError: If the agent does not exist.
        """
        if key not in self._nodes:
            raise KeyError(f"Agent '{key}' not found in graph")
        return dict(self._nodes[key])

    def all_agents(self) -> List[str]:
        """Return all agent keys in the graph."""
        return list(self._nodes.keys())

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_dependency(
        self,
        from_key: str,
        to_key: str,
        version_constraint: str = "*",
        edge_type: EdgeType = EdgeType.RUNTIME,
    ) -> DependencyEdge:
        """Add a dependency edge from one agent to another.

        Args:
            from_key: Agent that depends (consumer).
            to_key: Agent that is depended upon (provider).
            version_constraint: Semver range constraint.
            edge_type: Type of dependency relationship.

        Returns:
            The created DependencyEdge.

        Raises:
            KeyError: If either agent does not exist in the graph.
        """
        if from_key not in self._nodes:
            raise KeyError(f"Agent '{from_key}' not found in graph")
        if to_key not in self._nodes:
            raise KeyError(f"Agent '{to_key}' not found in graph")

        # Prevent duplicate edges
        for existing in self._edges[from_key]:
            if existing.to_key == to_key and existing.edge_type == edge_type:
                logger.debug("Dependency already exists: %s -> %s", from_key, to_key)
                return existing

        edge = DependencyEdge(
            from_key=from_key,
            to_key=to_key,
            version_constraint=version_constraint,
            edge_type=edge_type,
        )
        self._edges[from_key].append(edge)
        self._reverse.setdefault(to_key, []).append(edge)
        logger.debug("Added dependency: %s -> %s (%s)", from_key, to_key, edge_type.value)
        return edge

    def remove_dependency(self, from_key: str, to_key: str) -> bool:
        """Remove a dependency edge between two agents.

        Removes all edge types between the two agents.

        Args:
            from_key: Source agent.
            to_key: Target agent.

        Returns:
            True if at least one edge was removed.
        """
        removed = False
        edges_to_remove = [
            e for e in self._edges.get(from_key, []) if e.to_key == to_key
        ]
        for edge in edges_to_remove:
            self._edges[from_key].remove(edge)
            if edge in self._reverse.get(to_key, []):
                self._reverse[to_key].remove(edge)
            removed = True

        if removed:
            logger.debug("Removed dependency: %s -> %s", from_key, to_key)
        return removed

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def get_dependencies(self, key: str) -> List[str]:
        """Return direct dependencies of an agent (agents it depends on).

        Args:
            key: Agent key.

        Returns:
            List of agent keys this agent depends on.
        """
        return [edge.to_key for edge in self._edges.get(key, [])]

    def get_dependency_edges(self, key: str) -> List[DependencyEdge]:
        """Return all outgoing dependency edges for an agent.

        Args:
            key: Agent key.

        Returns:
            List of DependencyEdge objects.
        """
        return list(self._edges.get(key, []))

    def get_dependents(self, key: str) -> List[str]:
        """Return agents that depend on this one.

        Args:
            key: Agent key.

        Returns:
            List of agent keys that depend on this agent.
        """
        return [edge.from_key for edge in self._reverse.get(key, [])]

    def get_all_transitive_dependencies(self, key: str) -> Set[str]:
        """Return the full transitive closure of dependencies.

        Uses BFS to find all direct and indirect dependencies.

        Args:
            key: Agent key.

        Returns:
            Set of all agent keys in the dependency closure (excluding key itself).
        """
        visited: Set[str] = set()
        queue: List[str] = list(self.get_dependencies(key))

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self.get_dependencies(current))

        return visited

    def get_all_transitive_dependents(self, key: str) -> Set[str]:
        """Return all agents that transitively depend on this one.

        Args:
            key: Agent key.

        Returns:
            Set of agent keys that directly or indirectly depend on this agent.
        """
        visited: Set[str] = set()
        queue: List[str] = list(self.get_dependents(key))

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self.get_dependents(current))

        return visited

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary for persistence.

        Returns:
            Dict with nodes and edges lists.
        """
        return {
            "nodes": {
                key: dict(meta) for key, meta in self._nodes.items()
            },
            "edges": [
                {
                    "from_key": edge.from_key,
                    "to_key": edge.to_key,
                    "version_constraint": edge.version_constraint,
                    "edge_type": edge.edge_type.value,
                }
                for edges in self._edges.values()
                for edge in edges
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DependencyGraph:
        """Deserialize a graph from a dictionary.

        Args:
            data: Serialized graph data.

        Returns:
            Reconstructed DependencyGraph.
        """
        graph = cls()
        for key, meta in data.get("nodes", {}).items():
            graph.add_agent(key, meta)
        for edge_data in data.get("edges", []):
            graph.add_dependency(
                from_key=edge_data["from_key"],
                to_key=edge_data["to_key"],
                version_constraint=edge_data.get("version_constraint", "*"),
                edge_type=EdgeType(edge_data.get("edge_type", "runtime")),
            )
        return graph
