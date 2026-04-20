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

import json
import logging
import sqlite3
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from greenlang.entity_graph.models import EntityEdge, EntityNode
from greenlang.entity_graph.types import NodeType
from greenlang.schemas.base import new_uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_nodes (
    node_id     TEXT PRIMARY KEY,
    graph_id    TEXT NOT NULL,
    node_type   TEXT NOT NULL,
    name        TEXT NOT NULL,
    geography   TEXT,
    attributes  TEXT NOT NULL DEFAULT '{}',
    deleted_at  TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_node_type ON entity_nodes (node_type);
CREATE INDEX IF NOT EXISTS idx_node_graph ON entity_nodes (graph_id);
CREATE INDEX IF NOT EXISTS idx_node_alive ON entity_nodes (deleted_at);

CREATE TABLE IF NOT EXISTS entity_edges (
    edge_id     TEXT PRIMARY KEY,
    graph_id    TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    attributes  TEXT NOT NULL DEFAULT '{}',
    valid_from  TEXT,
    valid_to    TEXT,
    deleted_at  TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edge_source ON entity_edges (source_id);
CREATE INDEX IF NOT EXISTS idx_edge_target ON entity_edges (target_id);
CREATE INDEX IF NOT EXISTS idx_edge_type ON entity_edges (edge_type);
CREATE INDEX IF NOT EXISTS idx_edge_graph ON entity_edges (graph_id);
CREATE INDEX IF NOT EXISTS idx_edge_alive ON entity_edges (deleted_at);
"""


class _SQLiteGraphBackend:
    """Append + soft-delete SQLite persistence for the Entity Graph."""

    def __init__(self, sqlite_path: Union[str, Path]) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.sqlite_path),
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=OFF")
        self._conn.executescript(_SQLITE_SCHEMA)

    # ---------------- nodes

    def upsert_node(self, graph_id: str, node: EntityNode) -> None:
        now = datetime.now(timezone.utc).isoformat()
        created = (node.created_at.isoformat() if node.created_at else now)
        updated = (node.updated_at.isoformat() if node.updated_at else now)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO entity_nodes (
                    node_id, graph_id, node_type, name, geography,
                    attributes, deleted_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    graph_id   = excluded.graph_id,
                    node_type  = excluded.node_type,
                    name       = excluded.name,
                    geography  = excluded.geography,
                    attributes = excluded.attributes,
                    deleted_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    node.node_id,
                    graph_id,
                    node.node_type,
                    node.name,
                    node.geography,
                    json.dumps(node.attributes or {}, sort_keys=True, default=str),
                    created,
                    updated,
                ),
            )

    def soft_delete_node(self, node_id: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE entity_nodes SET deleted_at=? WHERE node_id=? AND deleted_at IS NULL",
                (now, node_id),
            )
            return cur.rowcount > 0

    def hard_delete_node(self, node_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM entity_nodes WHERE node_id=?",
                (node_id,),
            )
            return cur.rowcount > 0

    def list_nodes(self, graph_id: str, include_deleted: bool = False) -> list[dict]:
        sql = (
            "SELECT node_id, graph_id, node_type, name, geography, attributes, "
            "deleted_at, created_at, updated_at FROM entity_nodes "
            "WHERE graph_id=?"
        )
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        sql += " ORDER BY created_at ASC"
        with self._lock:
            rows = list(self._conn.execute(sql, (graph_id,)))
        return [self._row_to_node(r) for r in rows]

    # ---------------- edges

    def upsert_edge(self, graph_id: str, edge: EntityEdge) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO entity_edges (
                    edge_id, graph_id, source_id, target_id, edge_type,
                    weight, attributes, valid_from, valid_to,
                    deleted_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                ON CONFLICT(edge_id) DO UPDATE SET
                    graph_id   = excluded.graph_id,
                    source_id  = excluded.source_id,
                    target_id  = excluded.target_id,
                    edge_type  = excluded.edge_type,
                    weight     = excluded.weight,
                    attributes = excluded.attributes,
                    valid_from = excluded.valid_from,
                    valid_to   = excluded.valid_to,
                    deleted_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    edge.edge_id,
                    graph_id,
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type,
                    edge.weight,
                    json.dumps(edge.attributes or {}, sort_keys=True, default=str),
                    edge.valid_from.isoformat() if edge.valid_from else None,
                    edge.valid_to.isoformat() if edge.valid_to else None,
                    now,
                    now,
                ),
            )

    def soft_delete_edge(self, edge_id: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE entity_edges SET deleted_at=? WHERE edge_id=? AND deleted_at IS NULL",
                (now, edge_id),
            )
            return cur.rowcount > 0

    def list_edges(self, graph_id: str, include_deleted: bool = False) -> list[dict]:
        sql = (
            "SELECT edge_id, graph_id, source_id, target_id, edge_type, weight, "
            "attributes, valid_from, valid_to, deleted_at, created_at, updated_at "
            "FROM entity_edges WHERE graph_id=?"
        )
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        sql += " ORDER BY created_at ASC"
        with self._lock:
            rows = list(self._conn.execute(sql, (graph_id,)))
        return [self._row_to_edge(r) for r in rows]

    # ---------------- lifecycle

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------- helpers

    @staticmethod
    def _row_to_node(row: tuple) -> dict:
        (
            node_id, graph_id, node_type, name, geography,
            attributes, deleted_at, created_at, updated_at,
        ) = row
        return {
            "node_id": node_id,
            "graph_id": graph_id,
            "node_type": node_type,
            "name": name,
            "geography": geography,
            "attributes": json.loads(attributes) if attributes else {},
            "deleted_at": deleted_at,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    @staticmethod
    def _row_to_edge(row: tuple) -> dict:
        (
            edge_id, graph_id, source_id, target_id, edge_type, weight,
            attributes, valid_from, valid_to,
            deleted_at, created_at, updated_at,
        ) = row
        return {
            "edge_id": edge_id,
            "graph_id": graph_id,
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type,
            "weight": weight,
            "attributes": json.loads(attributes) if attributes else {},
            "valid_from": valid_from,
            "valid_to": valid_to,
            "deleted_at": deleted_at,
            "created_at": created_at,
            "updated_at": updated_at,
        }


_SUPPORTED_BACKENDS = {"memory", "sqlite"}


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

    def __init__(
        self,
        graph_id: str = "default",
        storage_backend: str = "memory",
        sqlite_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize an EntityGraph.

        Args:
            graph_id: Unique identifier for this graph instance.
            storage_backend: ``"memory"`` (default) or ``"sqlite"``.
            sqlite_path: Required when ``storage_backend == "sqlite"``.
        """
        if storage_backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                "Unsupported storage_backend %r; choose from %s"
                % (storage_backend, sorted(_SUPPORTED_BACKENDS))
            )

        self.graph_id: str = graph_id
        self.storage_backend: str = storage_backend
        self._nodes: dict[str, EntityNode] = {}
        self._edges: dict[str, EntityEdge] = {}
        self._outgoing: dict[str, list[str]] = defaultdict(list)
        self._incoming: dict[str, list[str]] = defaultdict(list)

        self.sqlite_backend: Optional[_SQLiteGraphBackend] = None
        if storage_backend == "sqlite":
            if sqlite_path is None:
                raise ValueError("storage_backend='sqlite' requires sqlite_path")
            self.sqlite_backend = _SQLiteGraphBackend(sqlite_path)
            # Hydrate in-memory structures from SQLite so queries are fast.
            self._hydrate_from_sqlite()

        logger.info(
            "EntityGraph '%s' initialized (backend=%s)",
            graph_id,
            storage_backend,
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: EntityNode, *, validate_type: bool = True) -> str:
        """Add or upsert a node to the graph.

        Args:
            node: The node to add.
            validate_type: When True (default), the ``node_type`` must be
                one of the canonical ``NodeType.ALL`` values.  Set False
                for free-form prototyping.

        Returns:
            The ``node_id`` of the added node.

        Raises:
            ValueError: If ``validate_type`` is True and ``node.node_type``
                is not a recognised ``NodeType``.
        """
        if not node.node_id:
            node = node.model_copy(update={"node_id": new_uuid()})
        if validate_type and not NodeType.is_valid(node.node_type):
            raise ValueError(
                "Unknown node_type %r; expected one of %s"
                % (node.node_type, NodeType.ALL)
            )
        self._nodes[node.node_id] = node
        if self.sqlite_backend is not None:
            self.sqlite_backend.upsert_node(self.graph_id, node)
        logger.debug("Node added: %s (%s)", node.node_id, node.node_type)
        return node.node_id

    def update_node(
        self,
        node_id: str,
        *,
        name: Optional[str] = None,
        geography: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> EntityNode:
        """Update mutable fields on an existing node.

        Returns the updated ``EntityNode``.  Raises ``KeyError`` if the
        node does not exist.
        """
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node '{node_id}' not found")
        updates: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
        if name is not None:
            updates["name"] = name
        if geography is not None:
            updates["geography"] = geography
        if attributes is not None:
            merged = dict(node.attributes or {})
            merged.update(attributes)
            updates["attributes"] = merged
        new_node = node.model_copy(update=updates)
        self._nodes[node_id] = new_node
        if self.sqlite_backend is not None:
            self.sqlite_backend.upsert_node(self.graph_id, new_node)
        return new_node

    def delete_node(self, node_id: str, *, soft: bool = True) -> bool:
        """Remove a node.

        ``soft=True`` (default) keeps the row in SQLite with a
        ``deleted_at`` timestamp so audit history is preserved.  In-memory
        adjacency is always purged.  ``soft=False`` hard-deletes (SQLite
        only) — in-memory behaviour is the same either way.

        Returns True if the node existed.
        """
        existed = node_id in self._nodes
        self._nodes.pop(node_id, None)
        # Purge adjacency entries pointing at this node.
        for eid in list(self._outgoing.get(node_id, [])):
            self._edges.pop(eid, None)
        for eid in list(self._incoming.get(node_id, [])):
            self._edges.pop(eid, None)
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)
        # Also remove any residual edges that still reference this node.
        for edge_id, edge in list(self._edges.items()):
            if edge.source_id == node_id or edge.target_id == node_id:
                self._edges.pop(edge_id, None)

        if self.sqlite_backend is not None:
            if soft:
                self.sqlite_backend.soft_delete_node(node_id)
            else:
                self.sqlite_backend.hard_delete_node(node_id)
        return existed

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
        if self.sqlite_backend is not None:
            self.sqlite_backend.upsert_edge(self.graph_id, edge)
        logger.debug(
            "Edge added: %s  %s -[%s]-> %s",
            edge.edge_id,
            edge.source_id,
            edge.edge_type,
            edge.target_id,
        )
        return edge.edge_id

    def delete_edge(self, edge_id: str, *, soft: bool = True) -> bool:
        """Remove an edge.  Mirror of ``delete_node``."""
        existed = edge_id in self._edges
        edge = self._edges.pop(edge_id, None)
        if edge is not None:
            if edge_id in self._outgoing.get(edge.source_id, []):
                self._outgoing[edge.source_id].remove(edge_id)
            if edge_id in self._incoming.get(edge.target_id, []):
                self._incoming[edge.target_id].remove(edge_id)
        if self.sqlite_backend is not None:
            if soft:
                self.sqlite_backend.soft_delete_edge(edge_id)
        return existed

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _hydrate_from_sqlite(self) -> None:
        """Load non-deleted nodes and edges from SQLite into memory."""
        if self.sqlite_backend is None:
            return
        for row in self.sqlite_backend.list_nodes(self.graph_id):
            try:
                created = (
                    datetime.fromisoformat(row["created_at"])
                    if row["created_at"] else None
                )
                updated = (
                    datetime.fromisoformat(row["updated_at"])
                    if row["updated_at"] else None
                )
                node = EntityNode(
                    node_id=row["node_id"],
                    node_type=row["node_type"],
                    name=row["name"],
                    geography=row["geography"],
                    attributes=row["attributes"] or {},
                    created_at=created,
                    updated_at=updated,
                )
                self._nodes[node.node_id] = node
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipped malformed node during hydrate: %s", exc)
        for row in self.sqlite_backend.list_edges(self.graph_id):
            try:
                vf = (
                    datetime.fromisoformat(row["valid_from"])
                    if row["valid_from"] else None
                )
                vt = (
                    datetime.fromisoformat(row["valid_to"])
                    if row["valid_to"] else None
                )
                edge = EntityEdge(
                    edge_id=row["edge_id"],
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    edge_type=row["edge_type"],
                    weight=row["weight"],
                    attributes=row["attributes"] or {},
                    valid_from=vf,
                    valid_to=vt,
                )
                self._edges[edge.edge_id] = edge
                self._outgoing[edge.source_id].append(edge.edge_id)
                self._incoming[edge.target_id].append(edge.edge_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipped malformed edge during hydrate: %s", exc)

    def close(self) -> None:
        """Close the SQLite connection (safe to call multiple times)."""
        if self.sqlite_backend is not None:
            self.sqlite_backend.close()
            self.sqlite_backend = None

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
