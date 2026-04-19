# -*- coding: utf-8 -*-
"""
Entity Graph Engine
====================

In-memory graph data structure for the v3 Entity Graph product module
(L1 Data Foundation).  Provides O(1) node/edge lookup, adjacency-list
traversal, BFS subgraph extraction, and filtered search.

This implementation is intentionally simple -- a production backend
(e.g. Neo4j, Neptune, or pg-graph) can replace the storage layer
without changing the public API.

Usage::

    from greenlang.entity_graph.graph import EntityGraph
    from greenlang.entity_graph.models import EntityNode, EntityEdge
    from greenlang.entity_graph.types import NodeType, EdgeType

    g = EntityGraph()
    g.add_node(EntityNode(node_id="org_1", node_type=NodeType.ORGANIZATION, name="Acme"))
    g.add_node(EntityNode(node_id="fac_1", node_type=NodeType.FACILITY, name="Berlin"))
    g.add_edge(EntityEdge(edge_id="e_1", source_id="org_1", target_id="fac_1", edge_type=EdgeType.OWNS))

    neighbors = g.get_neighbors("org_1")

Author: GreenLang Platform Team
Date: April 2026
Status: v3 Stub
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Optional

from greenlang.entity_graph.models import EntityEdge, EntityNode
from greenlang.schemas.base import new_uuid

logger = logging.getLogger(__name__)


class EntityGraph:
    """In-memory entity graph for the v3 Entity Graph product.

    Stores nodes and edges in dictionaries keyed by their IDs, with
    adjacency lists for fast outgoing / incoming edge traversal.

    Attributes:
        graph_id: Identifier for this graph instance.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, graph_id: str = "default") -> None:
        """Initialize an empty EntityGraph.

        Args:
            graph_id: Unique identifier for this graph instance.
        """
        self.graph_id: str = graph_id
        self._nodes: dict[str, EntityNode] = {}
        self._edges: dict[str, EntityEdge] = {}
        # adjacency: node_id -> list[edge_id]
        self._outgoing: dict[str, list[str]] = defaultdict(list)
        self._incoming: dict[str, list[str]] = defaultdict(list)
        logger.info("EntityGraph '%s' initialized", graph_id)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: EntityNode) -> str:
        """Add a node to the graph.

        If ``node.node_id`` is already present the existing node is
        silently overwritten (upsert semantics).

        Args:
            node: The node to add.

        Returns:
            The ``node_id`` of the added node.
        """
        if not node.node_id:
            node = node.model_copy(update={"node_id": new_uuid()})
        self._nodes[node.node_id] = node
        logger.debug("Node added: %s (%s)", node.node_id, node.node_type)
        return node.node_id

    def add_edge(self, edge: EntityEdge) -> str:
        """Add a directed edge to the graph.

        Both ``source_id`` and ``target_id`` must reference nodes that
        have already been added.  Raises ``ValueError`` otherwise.

        Args:
            edge: The edge to add.

        Returns:
            The ``edge_id`` of the added edge.

        Raises:
            ValueError: If source or target node does not exist.
        """
        if not edge.edge_id:
            edge = edge.model_copy(update={"edge_id": new_uuid()})

        if edge.source_id not in self._nodes:
            raise ValueError(
                "Source node '%s' not found in graph" % edge.source_id
            )
        if edge.target_id not in self._nodes:
            raise ValueError(
                "Target node '%s' not found in graph" % edge.target_id
            )

        self._edges[edge.edge_id] = edge
        self._outgoing[edge.source_id].append(edge.edge_id)
        self._incoming[edge.target_id].append(edge.edge_id)
        logger.debug(
            "Edge added: %s  %s -[%s]-> %s",
            edge.edge_id,
            edge.source_id,
            edge.edge_type,
            edge.target_id,
        )
        return edge.edge_id

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[EntityNode]:
        """Return a node by ID, or ``None`` if not found.

        Args:
            node_id: The node identifier to look up.

        Returns:
            The matching ``EntityNode`` or ``None``.
        """
        return self._nodes.get(node_id)

    def get_edges(
        self, node_id: str, direction: str = "outgoing"
    ) -> list[EntityEdge]:
        """Return edges connected to a node.

        Args:
            node_id: The node to query.
            direction: ``"outgoing"`` (default), ``"incoming"``, or
                ``"both"``.

        Returns:
            List of matching ``EntityEdge`` objects.

        Raises:
            ValueError: If *direction* is not one of the allowed values.
        """
        if direction == "outgoing":
            edge_ids = self._outgoing.get(node_id, [])
        elif direction == "incoming":
            edge_ids = self._incoming.get(node_id, [])
        elif direction == "both":
            edge_ids = list(
                dict.fromkeys(
                    self._outgoing.get(node_id, [])
                    + self._incoming.get(node_id, [])
                )
            )
        else:
            raise ValueError(
                "direction must be 'outgoing', 'incoming', or 'both', "
                "got '%s'" % direction
            )

        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
    ) -> list[EntityNode]:
        """Return neighbor nodes reachable via outgoing edges.

        Args:
            node_id: The source node.
            edge_type: If provided, only follow edges of this type.

        Returns:
            List of neighbor ``EntityNode`` objects (deduplicated,
            preserving insertion order).
        """
        edges = self.get_edges(node_id, direction="outgoing")
        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]

        seen: set[str] = set()
        neighbors: list[EntityNode] = []
        for edge in edges:
            tid = edge.target_id
            if tid not in seen:
                seen.add(tid)
                node = self._nodes.get(tid)
                if node is not None:
                    neighbors.append(node)
        return neighbors

    def find_nodes(
        self,
        node_type: Optional[str] = None,
        geography: Optional[str] = None,
        name_contains: Optional[str] = None,
    ) -> list[EntityNode]:
        """Search nodes by optional filters (AND logic).

        Args:
            node_type: Filter by ``node_type`` (exact match).
            geography: Filter by ``geography`` (exact match).
            name_contains: Filter by substring in ``name``
                (case-insensitive).

        Returns:
            List of matching ``EntityNode`` objects.
        """
        results: list[EntityNode] = []
        for node in self._nodes.values():
            if node_type is not None and node.node_type != node_type:
                continue
            if geography is not None and node.geography != geography:
                continue
            if name_contains is not None and name_contains.lower() not in node.name.lower():
                continue
            results.append(node)
        return results

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_subgraph(self, root_id: str, depth: int = 2) -> dict[str, Any]:
        """Extract a subgraph via BFS from *root_id*.

        Args:
            root_id: Starting node ID.
            depth: Maximum BFS depth (default 2).

        Returns:
            Dictionary with keys ``"nodes"`` (list of node dicts) and
            ``"edges"`` (list of edge dicts) representing the subgraph.

        Raises:
            ValueError: If *root_id* does not exist in the graph.
        """
        if root_id not in self._nodes:
            raise ValueError("Root node '%s' not found in graph" % root_id)

        visited_nodes: dict[str, EntityNode] = {}
        visited_edges: dict[str, EntityEdge] = {}
        queue: deque[tuple[str, int]] = deque([(root_id, 0)])

        while queue:
            current_id, current_depth = queue.popleft()
            if current_id in visited_nodes:
                continue
            node = self._nodes.get(current_id)
            if node is None:
                continue
            visited_nodes[current_id] = node

            if current_depth >= depth:
                continue

            for edge_id in self._outgoing.get(current_id, []):
                edge = self._edges.get(edge_id)
                if edge is None:
                    continue
                visited_edges[edge_id] = edge
                if edge.target_id not in visited_nodes:
                    queue.append((edge.target_id, current_depth + 1))

        return {
            "nodes": [n.model_dump() for n in visited_nodes.values()],
            "edges": [e.model_dump() for e in visited_edges.values()],
        }

    # ------------------------------------------------------------------
    # Serialization / Introspection
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full graph to a plain dictionary.

        Returns:
            Dictionary with ``graph_id``, ``nodes``, and ``edges`` keys.
        """
        return {
            "graph_id": self.graph_id,
            "nodes": [n.model_dump() for n in self._nodes.values()],
            "edges": [e.model_dump() for e in self._edges.values()],
        }

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the graph.

        Returns:
            Dictionary with ``node_count``, ``edge_count``,
            ``node_types`` (unique set), and ``edge_types`` (unique set).
        """
        node_types = sorted({n.node_type for n in self._nodes.values()})
        edge_types = sorted({e.edge_type for e in self._edges.values()})
        return {
            "graph_id": self.graph_id,
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "node_types": node_types,
            "edge_types": edge_types,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            "EntityGraph(graph_id='%s', nodes=%d, edges=%d)"
            % (self.graph_id, len(self._nodes), len(self._edges))
        )
