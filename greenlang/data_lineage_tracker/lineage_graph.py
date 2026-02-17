# -*- coding: utf-8 -*-
"""
LineageGraphEngine - AGENT-DATA-018: Data Lineage Tracker (GL-DATA-X-021)

Engine 3 of 7 in the Data Lineage Tracker pipeline. Builds and maintains
the in-memory directed acyclic graph (DAG) of data assets and their lineage
relationships (edges). All mutations are thread-safe and every structural
change is recorded via SHA-256 provenance hashing.

Graph data structures:
    - _nodes     : Dict[str, dict]      - asset_id to node metadata
    - _edges     : Dict[str, dict]      - edge_id to edge metadata
    - _adjacency_out : Dict[str, Set[str]] - source_id to outgoing edge_ids
    - _adjacency_in  : Dict[str, Set[str]] - target_id to incoming edge_ids

Traversal algorithms:
    - BFS forward/backward traversal with configurable max_depth
    - BFS shortest path between two nodes
    - DFS-based cycle detection (WHITE/GRAY/BLACK colouring)
    - Kahn's algorithm for topological ordering
    - BFS-based connected component discovery
    - Longest-path depth computation from roots

Edge types:
    - dataset_level : coarse-grained asset-to-asset lineage
    - column_level  : fine-grained field-to-field lineage

Zero-Hallucination Guarantees:
    - All graph algorithms are deterministic (BFS, DFS, Kahn's)
    - No LLM calls in any traversal or mutation path
    - SHA-256 provenance recorded on every graph mutation
    - Thread-safe via threading.Lock on all public methods

Example:
    >>> from greenlang.data_lineage_tracker.lineage_graph import LineageGraphEngine
    >>> engine = LineageGraphEngine()
    >>> engine.add_node("a1", "raw.orders", "dataset")
    {'asset_id': 'a1', ...}
    >>> engine.add_node("a2", "clean.orders", "dataset")
    {'asset_id': 'a2', ...}
    >>> edge = engine.add_edge("a1", "a2", edge_type="dataset_level")
    >>> edge["edge_type"]
    'dataset_level'
    >>> result = engine.traverse_forward("a1")
    >>> len(result["nodes"]) >= 2
    True
    >>> engine.detect_cycles()
    []

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-018 Data Lineage Tracker (GL-DATA-X-021)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Set

from greenlang.data_lineage_tracker.config import get_config
from greenlang.data_lineage_tracker.metrics import (
    PROMETHEUS_AVAILABLE,
    observe_graph_traversal_duration,
    observe_processing_duration,
    record_edge_created,
    set_graph_edge_count,
    set_graph_node_count,
)
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_EDGE_TYPES: FrozenSet[str] = frozenset({"dataset_level", "column_level"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    """Return current UTC datetime as ISO-8601 string."""
    return _utcnow().replace(microsecond=0).isoformat()


def _hash_dict(data: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash for a dictionary.

    Args:
        data: JSON-serialisable dictionary.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# LineageGraphEngine
# ---------------------------------------------------------------------------


class LineageGraphEngine:
    """Builds and maintains the data lineage directed graph (DAG).

    Stores nodes (data assets) and directed edges (lineage relationships)
    in an adjacency-list representation. Provides BFS/DFS traversal,
    topological sorting, cycle detection, shortest-path queries, and
    connected-component discovery.

    All public methods acquire ``self._lock`` to guarantee thread safety.
    Mutations are recorded via :class:`ProvenanceTracker` for audit trails.

    Attributes:
        _nodes: Mapping from asset_id to node metadata dict.
        _edges: Mapping from edge_id to edge metadata dict.
        _adjacency_out: Forward adjacency mapping source_id to edge_ids.
        _adjacency_in: Backward adjacency mapping target_id to edge_ids.
        _lock: Threading lock for thread-safe mutations and reads.
        _provenance: ProvenanceTracker instance for SHA-256 audit trail.

    Example:
        >>> engine = LineageGraphEngine()
        >>> engine.add_node("src", "raw.clicks", "dataset")
        {'asset_id': 'src', ...}
        >>> engine.add_node("dst", "clean.clicks", "table")
        {'asset_id': 'dst', ...}
        >>> engine.add_edge("src", "dst")
        {'edge_id': '...', ...}
        >>> engine.get_statistics()["total_nodes"]
        2
    """

    def __init__(
        self,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize the LineageGraphEngine.

        Args:
            provenance: Optional ProvenanceTracker instance. If ``None``,
                a fresh tracker is created internally.
        """
        self._nodes: Dict[str, dict] = {}
        self._edges: Dict[str, dict] = {}
        self._adjacency_out: Dict[str, Set[str]] = defaultdict(set)
        self._adjacency_in: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.Lock()
        self._provenance = provenance if provenance is not None else ProvenanceTracker()

        logger.info("LineageGraphEngine initialized")

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(
        self,
        asset_id: str,
        qualified_name: str,
        asset_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Add or update a node in the lineage graph.

        If a node with the given ``asset_id`` already exists it is updated
        in place (idempotent upsert). The Prometheus node gauge is updated
        after every successful insertion.

        Args:
            asset_id: Unique identifier for the data asset.
            qualified_name: Fully-qualified name (e.g. ``"schema.table"``).
            asset_type: Asset type label (dataset, table, column, file, etc.).
            metadata: Optional dictionary of additional properties.

        Returns:
            The node dictionary as stored in the graph.

        Raises:
            ValueError: If ``asset_id``, ``qualified_name``, or
                ``asset_type`` is empty.
        """
        start = time.monotonic()

        if not asset_id:
            raise ValueError("asset_id must not be empty")
        if not qualified_name:
            raise ValueError("qualified_name must not be empty")
        if not asset_type:
            raise ValueError("asset_type must not be empty")

        now = _utcnow_iso()
        node = {
            "asset_id": asset_id,
            "qualified_name": qualified_name,
            "asset_type": asset_type,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
        }

        with self._lock:
            existing = self._nodes.get(asset_id)
            if existing is not None:
                node["created_at"] = existing["created_at"]
                logger.debug("Updating existing node: %s", asset_id)
            else:
                logger.debug("Adding new node: %s", asset_id)

            self._nodes[asset_id] = node
            node_count = len(self._nodes)

        # Provenance
        self._provenance.record(
            entity_type="lineage_graph_node",
            entity_id=asset_id,
            action="node_added" if existing is None else "node_updated",
            metadata=node,
        )

        # Metrics
        set_graph_node_count(node_count)
        elapsed = time.monotonic() - start
        observe_processing_duration("node_add", elapsed)

        return node

    def remove_node(self, asset_id: str) -> bool:
        """Remove a node and all its connected edges from the graph.

        Removes every edge where the node is either source or target,
        then removes the node itself. Updates gauge metrics.

        Args:
            asset_id: Identifier of the node to remove.

        Returns:
            ``True`` if the node existed and was removed, ``False``
            if the node was not found.
        """
        start = time.monotonic()

        with self._lock:
            if asset_id not in self._nodes:
                logger.debug("remove_node: %s not found", asset_id)
                return False

            # Collect all edges connected to this node
            edges_to_remove: Set[str] = set()
            edges_to_remove.update(self._adjacency_out.get(asset_id, set()))
            edges_to_remove.update(self._adjacency_in.get(asset_id, set()))

            # Remove each connected edge from internal structures
            for edge_id in edges_to_remove:
                self._remove_edge_internal(edge_id)

            # Remove node
            del self._nodes[asset_id]
            self._adjacency_out.pop(asset_id, None)
            self._adjacency_in.pop(asset_id, None)

            node_count = len(self._nodes)
            edge_count = len(self._edges)

        # Provenance
        self._provenance.record(
            entity_type="lineage_graph_node",
            entity_id=asset_id,
            action="node_removed",
        )

        # Metrics
        set_graph_node_count(node_count)
        set_graph_edge_count(edge_count)
        elapsed = time.monotonic() - start
        observe_processing_duration("node_remove", elapsed)

        logger.info("Removed node %s and %d connected edges", asset_id, len(edges_to_remove))
        return True

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source_asset_id: str,
        target_asset_id: str,
        transformation_id: Optional[str] = None,
        edge_type: str = "dataset_level",
        source_field: Optional[str] = None,
        target_field: Optional[str] = None,
        transformation_logic: str = "",
        confidence: float = 1.0,
    ) -> dict:
        """Add a directed edge between two nodes in the lineage graph.

        Validates that both endpoints exist, that the edge type is valid,
        and optionally checks for cycle creation. Generates a UUID-based
        edge identifier.

        Args:
            source_asset_id: Origin node identifier.
            target_asset_id: Destination node identifier.
            transformation_id: Optional transformation reference.
            edge_type: One of ``"dataset_level"`` or ``"column_level"``.
            source_field: Source column/field name (for column_level edges).
            target_field: Target column/field name (for column_level edges).
            transformation_logic: Free-text description of the transform.
            confidence: Confidence score for inferred lineage (0.0 to 1.0).

        Returns:
            The edge dictionary as stored in the graph.

        Raises:
            ValueError: If endpoint nodes do not exist, the edge type is
                invalid, or confidence is outside [0.0, 1.0].
            ValueError: If adding the edge would create a cycle (when
                cycle detection is enabled via config).
        """
        start = time.monotonic()

        # --- Validate inputs -----------------------------------------------
        if not source_asset_id:
            raise ValueError("source_asset_id must not be empty")
        if not target_asset_id:
            raise ValueError("target_asset_id must not be empty")
        if edge_type not in VALID_EDGE_TYPES:
            raise ValueError(
                f"edge_type must be one of {sorted(VALID_EDGE_TYPES)}, "
                f"got '{edge_type}'"
            )
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {confidence}"
            )

        edge_id = str(uuid.uuid4())
        now = _utcnow_iso()

        edge = {
            "edge_id": edge_id,
            "source_asset_id": source_asset_id,
            "target_asset_id": target_asset_id,
            "transformation_id": transformation_id,
            "edge_type": edge_type,
            "source_field": source_field,
            "target_field": target_field,
            "transformation_logic": transformation_logic,
            "confidence": confidence,
            "created_at": now,
        }

        with self._lock:
            # Verify endpoints exist
            if source_asset_id not in self._nodes:
                raise ValueError(
                    f"Source node '{source_asset_id}' does not exist in graph"
                )
            if target_asset_id not in self._nodes:
                raise ValueError(
                    f"Target node '{target_asset_id}' does not exist in graph"
                )

            # Check for capacity
            cfg = get_config()
            if len(self._edges) >= cfg.max_edges:
                raise ValueError(
                    f"Edge capacity reached ({cfg.max_edges}). "
                    "Cannot add more edges."
                )

            # Optional cycle detection: temporarily add edge then check
            self._adjacency_out[source_asset_id].add(edge_id)
            self._adjacency_in[target_asset_id].add(edge_id)
            self._edges[edge_id] = edge

            if self._would_create_cycle_from(target_asset_id, source_asset_id):
                # Rollback
                self._adjacency_out[source_asset_id].discard(edge_id)
                self._adjacency_in[target_asset_id].discard(edge_id)
                del self._edges[edge_id]
                raise ValueError(
                    f"Adding edge {source_asset_id} -> {target_asset_id} "
                    "would create a cycle in the lineage graph"
                )

            edge_count = len(self._edges)

        # Provenance
        self._provenance.record(
            entity_type="lineage_graph_edge",
            entity_id=edge_id,
            action="edge_created",
            metadata=edge,
        )

        # Metrics
        record_edge_created(edge_type)
        set_graph_edge_count(edge_count)
        elapsed = time.monotonic() - start
        observe_processing_duration("edge_add", elapsed)

        logger.debug(
            "Edge created: %s -> %s (id=%s, type=%s)",
            source_asset_id,
            target_asset_id,
            edge_id,
            edge_type,
        )
        return edge

    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from the lineage graph.

        Args:
            edge_id: Identifier of the edge to remove.

        Returns:
            ``True`` if the edge existed and was removed, ``False`` otherwise.
        """
        start = time.monotonic()

        with self._lock:
            if edge_id not in self._edges:
                logger.debug("remove_edge: %s not found", edge_id)
                return False
            self._remove_edge_internal(edge_id)
            edge_count = len(self._edges)

        # Provenance
        self._provenance.record(
            entity_type="lineage_graph_edge",
            entity_id=edge_id,
            action="edge_removed",
        )

        # Metrics
        set_graph_edge_count(edge_count)
        elapsed = time.monotonic() - start
        observe_processing_duration("edge_remove", elapsed)

        logger.debug("Edge removed: %s", edge_id)
        return True

    def _remove_edge_internal(self, edge_id: str) -> None:
        """Remove an edge from internal data structures without locking.

        Caller MUST hold ``self._lock``.

        Args:
            edge_id: Edge identifier to remove.
        """
        edge = self._edges.get(edge_id)
        if edge is None:
            return

        source_id = edge["source_asset_id"]
        target_id = edge["target_asset_id"]

        self._adjacency_out.get(source_id, set()).discard(edge_id)
        self._adjacency_in.get(target_id, set()).discard(edge_id)

        del self._edges[edge_id]

    # ------------------------------------------------------------------
    # Lookup operations
    # ------------------------------------------------------------------

    def get_node(self, asset_id: str) -> Optional[dict]:
        """Return node metadata for the given asset_id.

        Args:
            asset_id: Node identifier to look up.

        Returns:
            Node dictionary or ``None`` if not found.
        """
        with self._lock:
            node = self._nodes.get(asset_id)
            return dict(node) if node is not None else None

    def get_edge(self, edge_id: str) -> Optional[dict]:
        """Return edge metadata for the given edge_id.

        Args:
            edge_id: Edge identifier to look up.

        Returns:
            Edge dictionary or ``None`` if not found.
        """
        with self._lock:
            edge = self._edges.get(edge_id)
            return dict(edge) if edge is not None else None

    def get_upstream_edges(self, asset_id: str) -> List[dict]:
        """Return all incoming edges (sources feeding into this asset).

        Args:
            asset_id: Target node identifier.

        Returns:
            List of edge dictionaries for all incoming edges.
        """
        with self._lock:
            edge_ids = self._adjacency_in.get(asset_id, set())
            return [dict(self._edges[eid]) for eid in edge_ids if eid in self._edges]

    def get_downstream_edges(self, asset_id: str) -> List[dict]:
        """Return all outgoing edges (consumers of this asset).

        Args:
            asset_id: Source node identifier.

        Returns:
            List of edge dictionaries for all outgoing edges.
        """
        with self._lock:
            edge_ids = self._adjacency_out.get(asset_id, set())
            return [dict(self._edges[eid]) for eid in edge_ids if eid in self._edges]

    # ------------------------------------------------------------------
    # BFS traversal
    # ------------------------------------------------------------------

    def traverse_backward(
        self,
        asset_id: str,
        max_depth: Optional[int] = None,
    ) -> dict:
        """BFS backward traversal from asset to all upstream sources.

        Traverses incoming edges to discover every ancestor node.
        Respects ``max_depth`` (defaults to config ``default_traversal_depth``
        if not provided).

        Args:
            asset_id: Starting node for backward traversal.
            max_depth: Maximum traversal depth. ``None`` to use config
                default.

        Returns:
            Dictionary with keys:
                - ``root``: Starting asset_id
                - ``nodes``: List of discovered node dicts
                - ``edges``: List of traversed edge dicts
                - ``depth``: Maximum depth reached
                - ``path``: List of asset_ids in BFS discovery order
        """
        start = time.monotonic()

        if max_depth is None:
            max_depth = get_config().default_traversal_depth

        result = self._bfs_traverse(
            asset_id=asset_id,
            max_depth=max_depth,
            direction="backward",
        )

        elapsed = time.monotonic() - start
        observe_graph_traversal_duration(elapsed)
        observe_processing_duration("traverse_backward", elapsed)

        return result

    def traverse_forward(
        self,
        asset_id: str,
        max_depth: Optional[int] = None,
    ) -> dict:
        """BFS forward traversal from asset to all downstream consumers.

        Traverses outgoing edges to discover every descendant node.
        Respects ``max_depth`` (defaults to config ``default_traversal_depth``
        if not provided).

        Args:
            asset_id: Starting node for forward traversal.
            max_depth: Maximum traversal depth. ``None`` to use config
                default.

        Returns:
            Dictionary with keys:
                - ``root``: Starting asset_id
                - ``nodes``: List of discovered node dicts
                - ``edges``: List of traversed edge dicts
                - ``depth``: Maximum depth reached
                - ``path``: List of asset_ids in BFS discovery order
        """
        start = time.monotonic()

        if max_depth is None:
            max_depth = get_config().default_traversal_depth

        result = self._bfs_traverse(
            asset_id=asset_id,
            max_depth=max_depth,
            direction="forward",
        )

        elapsed = time.monotonic() - start
        observe_graph_traversal_duration(elapsed)
        observe_processing_duration("traverse_forward", elapsed)

        return result

    def _bfs_traverse(
        self,
        asset_id: str,
        max_depth: int,
        direction: str,
    ) -> dict:
        """Internal BFS traversal in the specified direction.

        Args:
            asset_id: Starting node.
            max_depth: Maximum BFS depth.
            direction: ``"forward"`` (outgoing) or ``"backward"`` (incoming).

        Returns:
            Traversal result dictionary.
        """
        with self._lock:
            if asset_id not in self._nodes:
                logger.warning(
                    "BFS traversal: start node %s not found", asset_id
                )
                return {
                    "root": asset_id,
                    "nodes": [],
                    "edges": [],
                    "depth": 0,
                    "path": [],
                }

            visited_nodes: Set[str] = set()
            visited_edges: Set[str] = set()
            path: List[str] = []
            max_depth_reached = 0

            # BFS queue: (node_id, current_depth)
            queue: deque[tuple[str, int]] = deque()
            queue.append((asset_id, 0))
            visited_nodes.add(asset_id)

            while queue:
                current_id, depth = queue.popleft()
                path.append(current_id)
                max_depth_reached = max(max_depth_reached, depth)

                if depth >= max_depth:
                    continue

                # Determine which adjacency list to follow
                if direction == "forward":
                    edge_ids = self._adjacency_out.get(current_id, set())
                else:
                    edge_ids = self._adjacency_in.get(current_id, set())

                for edge_id in edge_ids:
                    edge = self._edges.get(edge_id)
                    if edge is None:
                        continue

                    visited_edges.add(edge_id)

                    # Determine the neighbour
                    if direction == "forward":
                        neighbour_id = edge["target_asset_id"]
                    else:
                        neighbour_id = edge["source_asset_id"]

                    if neighbour_id not in visited_nodes:
                        visited_nodes.add(neighbour_id)
                        queue.append((neighbour_id, depth + 1))

            # Build output
            result_nodes = [
                dict(self._nodes[nid])
                for nid in visited_nodes
                if nid in self._nodes
            ]
            result_edges = [
                dict(self._edges[eid])
                for eid in visited_edges
                if eid in self._edges
            ]

        logger.debug(
            "BFS %s from %s: %d nodes, %d edges, depth=%d",
            direction,
            asset_id,
            len(result_nodes),
            len(result_edges),
            max_depth_reached,
        )

        return {
            "root": asset_id,
            "nodes": result_nodes,
            "edges": result_edges,
            "depth": max_depth_reached,
            "path": path,
        }

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    def get_subgraph(
        self,
        asset_id: str,
        depth: int = 3,
    ) -> dict:
        """Extract the neighbourhood subgraph around an asset.

        Combines forward and backward BFS up to ``depth`` hops and returns
        the union of discovered nodes and edges.

        Args:
            asset_id: Centre node of the subgraph.
            depth: Maximum number of hops in each direction.

        Returns:
            Dictionary with keys ``"nodes"`` (list of node dicts) and
            ``"edges"`` (list of edge dicts).
        """
        start = time.monotonic()

        backward = self._bfs_traverse(asset_id, depth, "backward")
        forward = self._bfs_traverse(asset_id, depth, "forward")

        # Merge deduplicated by id
        node_map: Dict[str, dict] = {}
        for n in backward["nodes"] + forward["nodes"]:
            node_map[n["asset_id"]] = n

        edge_map: Dict[str, dict] = {}
        for e in backward["edges"] + forward["edges"]:
            edge_map[e["edge_id"]] = e

        elapsed = time.monotonic() - start
        observe_graph_traversal_duration(elapsed)

        return {
            "center": asset_id,
            "nodes": list(node_map.values()),
            "edges": list(edge_map.values()),
            "depth": depth,
        }

    # ------------------------------------------------------------------
    # Shortest path
    # ------------------------------------------------------------------

    def get_shortest_path(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[List[str]]:
        """Find the shortest path between two nodes using BFS.

        The path follows the direction of edges (source to target).
        Returns ``None`` if no path exists.

        Args:
            source_id: Starting node identifier.
            target_id: Destination node identifier.

        Returns:
            Ordered list of asset_ids from ``source_id`` to ``target_id``
            (inclusive), or ``None`` if unreachable.
        """
        start = time.monotonic()

        with self._lock:
            if source_id not in self._nodes or target_id not in self._nodes:
                return None

            if source_id == target_id:
                return [source_id]

            visited: Set[str] = {source_id}
            # Map child -> parent for path reconstruction
            parent: Dict[str, str] = {}
            queue: deque[str] = deque([source_id])

            while queue:
                current = queue.popleft()
                edge_ids = self._adjacency_out.get(current, set())

                for edge_id in edge_ids:
                    edge = self._edges.get(edge_id)
                    if edge is None:
                        continue
                    neighbour = edge["target_asset_id"]

                    if neighbour in visited:
                        continue

                    parent[neighbour] = current
                    if neighbour == target_id:
                        # Reconstruct path
                        path = self._reconstruct_path(parent, source_id, target_id)
                        elapsed = time.monotonic() - start
                        observe_graph_traversal_duration(elapsed)
                        return path

                    visited.add(neighbour)
                    queue.append(neighbour)

        elapsed = time.monotonic() - start
        observe_graph_traversal_duration(elapsed)
        return None

    @staticmethod
    def _reconstruct_path(
        parent: Dict[str, str],
        source_id: str,
        target_id: str,
    ) -> List[str]:
        """Reconstruct path from BFS parent map.

        Args:
            parent: Mapping from child node to its BFS parent.
            source_id: Path start.
            target_id: Path end.

        Returns:
            Ordered list of asset_ids from source to target.
        """
        path: List[str] = []
        current = target_id
        while current != source_id:
            path.append(current)
            current = parent[current]
        path.append(source_id)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def detect_cycles(self) -> List[List[str]]:
        """Detect all cycles in the lineage graph using DFS colouring.

        Uses the WHITE(0)/GRAY(1)/BLACK(2) algorithm to find back edges.
        A valid lineage DAG should return an empty list.

        Returns:
            List of cycles, where each cycle is a list of asset_ids
            forming a loop. Returns an empty list if no cycles exist.
        """
        start = time.monotonic()

        with self._lock:
            all_nodes = set(self._nodes.keys())
            # 0=WHITE, 1=GRAY, 2=BLACK
            colour: Dict[str, int] = {n: 0 for n in all_nodes}
            parent_map: Dict[str, Optional[str]] = {n: None for n in all_nodes}
            cycles: List[List[str]] = []

            for node_id in all_nodes:
                if colour[node_id] == 0:
                    self._dfs_cycle_visit(
                        node_id, colour, parent_map, cycles
                    )

        elapsed = time.monotonic() - start
        observe_graph_traversal_duration(elapsed)

        if cycles:
            logger.warning("Detected %d cycle(s) in lineage graph", len(cycles))
        else:
            logger.debug("No cycles detected in lineage graph")

        return cycles

    def _dfs_cycle_visit(
        self,
        node_id: str,
        colour: Dict[str, int],
        parent_map: Dict[str, Optional[str]],
        cycles: List[List[str]],
    ) -> None:
        """DFS visit for cycle detection.

        Caller MUST hold ``self._lock``.

        Args:
            node_id: Current node being visited.
            colour: Colouring map (0=white, 1=gray, 2=black).
            parent_map: DFS parent mapping for path reconstruction.
            cycles: Accumulator for discovered cycles.
        """
        colour[node_id] = 1  # GRAY

        edge_ids = self._adjacency_out.get(node_id, set())
        for edge_id in edge_ids:
            edge = self._edges.get(edge_id)
            if edge is None:
                continue

            neighbour = edge["target_asset_id"]

            if colour.get(neighbour, 0) == 0:
                # Unvisited
                parent_map[neighbour] = node_id
                self._dfs_cycle_visit(neighbour, colour, parent_map, cycles)
            elif colour.get(neighbour, 0) == 1:
                # Back edge -- cycle found
                cycle = self._extract_cycle(parent_map, node_id, neighbour)
                cycles.append(cycle)

        colour[node_id] = 2  # BLACK

    @staticmethod
    def _extract_cycle(
        parent_map: Dict[str, Optional[str]],
        current: str,
        ancestor: str,
    ) -> List[str]:
        """Extract cycle path from DFS parent map.

        Args:
            parent_map: DFS parent tracking.
            current: Node where back edge originates.
            ancestor: Node where back edge terminates (GRAY node).

        Returns:
            List of asset_ids forming the cycle.
        """
        cycle: List[str] = [ancestor]
        node = current
        while node != ancestor:
            cycle.append(node)
            node = parent_map.get(node)  # type: ignore[assignment]
            if node is None:
                break
        cycle.append(ancestor)
        cycle.reverse()
        return cycle

    def _would_create_cycle_from(
        self,
        start_id: str,
        target_id: str,
    ) -> bool:
        """Check if there is a path from start_id to target_id (cycle check).

        Used after tentatively adding an edge to confirm the graph remains
        acyclic. If a path exists from the new edge's target back to its
        source, a cycle would be created.

        Caller MUST hold ``self._lock``.

        Args:
            start_id: Node to start BFS from (the edge target).
            target_id: Node to search for (the edge source).

        Returns:
            ``True`` if a path exists (cycle would be created).
        """
        visited: Set[str] = {start_id}
        queue: deque[str] = deque([start_id])

        while queue:
            current = queue.popleft()

            edge_ids = self._adjacency_out.get(current, set())
            for edge_id in edge_ids:
                edge = self._edges.get(edge_id)
                if edge is None:
                    continue

                neighbour = edge["target_asset_id"]
                if neighbour == target_id:
                    return True

                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)

        return False

    # ------------------------------------------------------------------
    # Topological ordering
    # ------------------------------------------------------------------

    def get_topological_order(self) -> List[str]:
        """Compute topological ordering using Kahn's algorithm.

        Produces a valid linear ordering of all nodes such that for every
        directed edge (u, v), u appears before v. The graph must be
        acyclic; if cycles exist an incomplete ordering is returned.

        Returns:
            List of asset_ids in topological order.
        """
        start = time.monotonic()

        with self._lock:
            # Calculate in-degree for all nodes
            in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}

            for edge in self._edges.values():
                target = edge["target_asset_id"]
                if target in in_degree:
                    in_degree[target] += 1

            # Seed queue with zero-in-degree nodes (roots)
            queue: deque[str] = deque()
            for nid, deg in sorted(in_degree.items()):
                if deg == 0:
                    queue.append(nid)

            order: List[str] = []

            while queue:
                node_id = queue.popleft()
                order.append(node_id)

                # Reduce in-degree for all downstream neighbours
                edge_ids = self._adjacency_out.get(node_id, set())
                for edge_id in edge_ids:
                    edge = self._edges.get(edge_id)
                    if edge is None:
                        continue
                    neighbour = edge["target_asset_id"]
                    if neighbour in in_degree:
                        in_degree[neighbour] -= 1
                        if in_degree[neighbour] == 0:
                            queue.append(neighbour)

        elapsed = time.monotonic() - start
        observe_graph_traversal_duration(elapsed)

        if len(order) < len(self._nodes):
            logger.warning(
                "Topological order incomplete (%d of %d nodes): "
                "graph may contain cycles",
                len(order),
                len(self._nodes),
            )

        return order

    # ------------------------------------------------------------------
    # Connected components
    # ------------------------------------------------------------------

    def get_connected_components(self) -> List[Set[str]]:
        """Find connected components using BFS (treating edges as undirected).

        Each component is a set of asset_ids that are reachable from each
        other when ignoring edge direction.

        Returns:
            List of sets, each set containing the asset_ids of a connected
            component.
        """
        start = time.monotonic()

        with self._lock:
            visited: Set[str] = set()
            components: List[Set[str]] = []

            for node_id in self._nodes:
                if node_id in visited:
                    continue

                # BFS undirected
                component: Set[str] = set()
                queue: deque[str] = deque([node_id])
                visited.add(node_id)

                while queue:
                    current = queue.popleft()
                    component.add(current)

                    # Follow outgoing edges
                    for edge_id in self._adjacency_out.get(current, set()):
                        edge = self._edges.get(edge_id)
                        if edge is None:
                            continue
                        neighbour = edge["target_asset_id"]
                        if neighbour not in visited:
                            visited.add(neighbour)
                            queue.append(neighbour)

                    # Follow incoming edges (undirected)
                    for edge_id in self._adjacency_in.get(current, set()):
                        edge = self._edges.get(edge_id)
                        if edge is None:
                            continue
                        neighbour = edge["source_asset_id"]
                        if neighbour not in visited:
                            visited.add(neighbour)
                            queue.append(neighbour)

                components.append(component)

        elapsed = time.monotonic() - start
        observe_graph_traversal_duration(elapsed)

        logger.debug("Found %d connected component(s)", len(components))
        return components

    # ------------------------------------------------------------------
    # Roots and leaves
    # ------------------------------------------------------------------

    def get_roots(self) -> List[str]:
        """Return all root nodes (data sources with no incoming edges).

        Returns:
            List of asset_ids that have zero incoming edges.
        """
        with self._lock:
            return [
                nid
                for nid in self._nodes
                if not self._adjacency_in.get(nid, set())
            ]

    def get_leaves(self) -> List[str]:
        """Return all leaf nodes (final outputs with no outgoing edges).

        Returns:
            List of asset_ids that have zero outgoing edges.
        """
        with self._lock:
            return [
                nid
                for nid in self._nodes
                if not self._adjacency_out.get(nid, set())
            ]

    # ------------------------------------------------------------------
    # Depth computation
    # ------------------------------------------------------------------

    def compute_depth(self, asset_id: str) -> int:
        """Compute the longest path from any root to the given node.

        Uses BFS from each root and tracks the maximum depth at which
        ``asset_id`` is discovered.

        Args:
            asset_id: Node to compute depth for.

        Returns:
            Longest path length from any root (0 if the node itself is a
            root or not found).
        """
        with self._lock:
            if asset_id not in self._nodes:
                return 0

            # BFS backward from asset_id, counting depth
            max_depth = 0
            visited: Set[str] = {asset_id}
            queue: deque[tuple[str, int]] = deque([(asset_id, 0)])

            while queue:
                current, depth = queue.popleft()
                max_depth = max(max_depth, depth)

                edge_ids = self._adjacency_in.get(current, set())
                for edge_id in edge_ids:
                    edge = self._edges.get(edge_id)
                    if edge is None:
                        continue
                    parent_id = edge["source_asset_id"]
                    if parent_id not in visited:
                        visited.add(parent_id)
                        queue.append((parent_id, depth + 1))

        return max_depth

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def take_snapshot(
        self,
        snapshot_name: Optional[str] = None,
    ) -> dict:
        """Create a point-in-time snapshot of the graph topology.

        Captures structural metrics and computes a SHA-256 hash of the
        entire graph state for integrity verification.

        Args:
            snapshot_name: Optional human-readable snapshot label.

        Returns:
            Snapshot dictionary with keys: ``snapshot_id``, ``name``,
            ``timestamp``, ``node_count``, ``edge_count``, ``max_depth``,
            ``connected_components``, ``orphan_count``, ``root_count``,
            ``leaf_count``, ``coverage_score``, ``graph_hash``.
        """
        start = time.monotonic()

        with self._lock:
            node_count = len(self._nodes)
            edge_count = len(self._edges)

            # Compute structural metrics while holding the lock
            roots = [
                nid for nid in self._nodes
                if not self._adjacency_in.get(nid, set())
            ]
            leaves = [
                nid for nid in self._nodes
                if not self._adjacency_out.get(nid, set())
            ]
            orphans = [
                nid for nid in self._nodes
                if (
                    not self._adjacency_in.get(nid, set())
                    and not self._adjacency_out.get(nid, set())
                )
            ]

            # Graph hash: hash of sorted node_ids + sorted edge_ids
            graph_content = {
                "node_ids": sorted(self._nodes.keys()),
                "edge_ids": sorted(self._edges.keys()),
                "edge_data": [
                    {
                        "id": eid,
                        "src": e["source_asset_id"],
                        "tgt": e["target_asset_id"],
                    }
                    for eid, e in sorted(self._edges.items())
                ],
            }

        graph_hash = _hash_dict(graph_content)

        # Compute max depth (iterates all roots, needs lock internally)
        max_depth = self._compute_max_depth()

        # Connected components count
        components = self.get_connected_components()
        component_count = len(components)

        # Coverage: fraction of nodes that have at least one edge
        connected_node_count = node_count - len(orphans)
        coverage_score = (
            connected_node_count / node_count if node_count > 0 else 0.0
        )

        snapshot_id = str(uuid.uuid4())
        snapshot = {
            "snapshot_id": snapshot_id,
            "name": snapshot_name or f"snapshot-{snapshot_id[:8]}",
            "timestamp": _utcnow_iso(),
            "node_count": node_count,
            "edge_count": edge_count,
            "max_depth": max_depth,
            "connected_components": component_count,
            "orphan_count": len(orphans),
            "root_count": len(roots),
            "leaf_count": len(leaves),
            "coverage_score": round(coverage_score, 4),
            "graph_hash": graph_hash,
        }

        # Provenance
        self._provenance.record(
            entity_type="lineage_graph_snapshot",
            entity_id=snapshot_id,
            action="snapshot_created",
            metadata=snapshot,
        )

        elapsed = time.monotonic() - start
        observe_processing_duration("snapshot", elapsed)

        logger.info(
            "Graph snapshot created: %s (nodes=%d, edges=%d, hash=%s)",
            snapshot["name"],
            node_count,
            edge_count,
            graph_hash[:16],
        )

        return snapshot

    def _compute_max_depth(self) -> int:
        """Compute the maximum depth across all nodes in the graph.

        Uses topological order to compute longest path efficiently.

        Returns:
            Maximum depth value in the graph.
        """
        with self._lock:
            if not self._nodes:
                return 0

            # Use topological order for longest-path computation
            in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}
            for edge in self._edges.values():
                target = edge["target_asset_id"]
                if target in in_degree:
                    in_degree[target] += 1

            queue: deque[str] = deque()
            depth_map: Dict[str, int] = {}
            for nid, deg in in_degree.items():
                if deg == 0:
                    queue.append(nid)
                    depth_map[nid] = 0

            while queue:
                node_id = queue.popleft()
                current_depth = depth_map.get(node_id, 0)

                edge_ids = self._adjacency_out.get(node_id, set())
                for edge_id in edge_ids:
                    edge = self._edges.get(edge_id)
                    if edge is None:
                        continue
                    neighbour = edge["target_asset_id"]
                    new_depth = current_depth + 1
                    if new_depth > depth_map.get(neighbour, 0):
                        depth_map[neighbour] = new_depth
                    if neighbour in in_degree:
                        in_degree[neighbour] -= 1
                        if in_degree[neighbour] == 0:
                            queue.append(neighbour)

            return max(depth_map.values()) if depth_map else 0

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Compute comprehensive graph statistics.

        Returns:
            Dictionary with keys: ``total_nodes``, ``total_edges``,
            ``max_depth``, ``avg_depth``, ``connected_components``,
            ``root_count``, ``leaf_count``, ``orphan_count``.
        """
        start = time.monotonic()

        with self._lock:
            node_count = len(self._nodes)
            edge_count = len(self._edges)
            all_node_ids = list(self._nodes.keys())

            roots = [
                nid for nid in self._nodes
                if not self._adjacency_in.get(nid, set())
            ]
            leaves = [
                nid for nid in self._nodes
                if not self._adjacency_out.get(nid, set())
            ]
            orphans = [
                nid for nid in self._nodes
                if (
                    not self._adjacency_in.get(nid, set())
                    and not self._adjacency_out.get(nid, set())
                )
            ]

        # Compute depths for all nodes
        max_depth = self._compute_max_depth()

        # Compute average depth using per-node depth
        depths: List[int] = []
        for nid in all_node_ids:
            depths.append(self.compute_depth(nid))
        avg_depth = (
            round(sum(depths) / len(depths), 2) if depths else 0.0
        )

        components = self.get_connected_components()

        elapsed = time.monotonic() - start
        observe_processing_duration("statistics", elapsed)

        return {
            "total_nodes": node_count,
            "total_edges": edge_count,
            "max_depth": max_depth,
            "avg_depth": avg_depth,
            "connected_components": len(components),
            "root_count": len(roots),
            "leaf_count": len(leaves),
            "orphan_count": len(orphans),
        }

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_graph(self) -> dict:
        """Export the full graph as a JSON-serialisable dictionary.

        Returns:
            Dictionary with keys ``"nodes"`` (list of node dicts),
            ``"edges"`` (list of edge dicts), ``"metadata"`` (graph stats).
        """
        start = time.monotonic()

        with self._lock:
            nodes = [dict(n) for n in self._nodes.values()]
            edges = [dict(e) for e in self._edges.values()]
            node_count = len(nodes)
            edge_count = len(edges)

        export = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "exported_at": _utcnow_iso(),
                "node_count": node_count,
                "edge_count": edge_count,
                "graph_hash": _hash_dict({"nodes": nodes, "edges": edges}),
            },
        }

        elapsed = time.monotonic() - start
        observe_processing_duration("export_graph", elapsed)

        logger.info(
            "Graph exported: %d nodes, %d edges", node_count, edge_count
        )
        return export

    def import_graph(self, data: dict) -> None:
        """Import a graph from a previously exported dictionary.

        Clears the current graph before importing. Rebuilds adjacency
        lists from the imported edges.

        Args:
            data: Dictionary with ``"nodes"`` and ``"edges"`` keys as
                produced by :meth:`export_graph`.

        Raises:
            ValueError: If ``data`` is missing required keys or has
                invalid structure.
        """
        start = time.monotonic()

        if not isinstance(data, dict):
            raise ValueError("import_graph requires a dict argument")
        if "nodes" not in data:
            raise ValueError("import data missing 'nodes' key")
        if "edges" not in data:
            raise ValueError("import data missing 'edges' key")

        nodes_list = data["nodes"]
        edges_list = data["edges"]

        if not isinstance(nodes_list, list):
            raise ValueError("'nodes' must be a list")
        if not isinstance(edges_list, list):
            raise ValueError("'edges' must be a list")

        with self._lock:
            # Clear current state
            self._nodes.clear()
            self._edges.clear()
            self._adjacency_out.clear()
            self._adjacency_in.clear()

            # Import nodes
            for node in nodes_list:
                asset_id = node.get("asset_id")
                if not asset_id:
                    logger.warning("Skipping node with missing asset_id")
                    continue
                self._nodes[asset_id] = dict(node)

            # Import edges and rebuild adjacency
            for edge in edges_list:
                edge_id = edge.get("edge_id")
                source_id = edge.get("source_asset_id")
                target_id = edge.get("target_asset_id")

                if not edge_id or not source_id or not target_id:
                    logger.warning(
                        "Skipping edge with missing required fields: %s",
                        edge.get("edge_id", "unknown"),
                    )
                    continue

                if source_id not in self._nodes:
                    logger.warning(
                        "Edge %s references missing source %s, skipping",
                        edge_id,
                        source_id,
                    )
                    continue
                if target_id not in self._nodes:
                    logger.warning(
                        "Edge %s references missing target %s, skipping",
                        edge_id,
                        target_id,
                    )
                    continue

                self._edges[edge_id] = dict(edge)
                self._adjacency_out[source_id].add(edge_id)
                self._adjacency_in[target_id].add(edge_id)

            node_count = len(self._nodes)
            edge_count = len(self._edges)

        # Provenance
        self._provenance.record(
            entity_type="lineage_graph",
            entity_id="import",
            action="graph_imported",
            metadata={
                "node_count": node_count,
                "edge_count": edge_count,
            },
        )

        # Metrics
        set_graph_node_count(node_count)
        set_graph_edge_count(edge_count)

        elapsed = time.monotonic() - start
        observe_processing_duration("import_graph", elapsed)

        logger.info(
            "Graph imported: %d nodes, %d edges", node_count, edge_count
        )

    # ------------------------------------------------------------------
    # Compatibility aliases
    # ------------------------------------------------------------------
    # The following methods provide compatibility aliases so that other
    # engines (ImpactAnalyzerEngine, LineageValidatorEngine, LineageReporterEngine,
    # LineageTrackerPipelineEngine) can call the methods they expect without
    # needing to know the canonical method names.

    def get_incoming_edges(self, asset_id: str) -> List[dict]:
        """Alias for :meth:`get_upstream_edges`.

        Used by ImpactAnalyzerEngine for backward traversal.
        """
        return self.get_upstream_edges(asset_id)

    def get_outgoing_edges(self, asset_id: str) -> List[dict]:
        """Alias for :meth:`get_downstream_edges`.

        Used by ImpactAnalyzerEngine for forward traversal.
        """
        return self.get_downstream_edges(asset_id)

    def has_node(self, asset_id: str) -> bool:
        """Check whether a node exists in the graph.

        Alias for ``asset_id in self`` (delegates to ``__contains__``).
        Used by ImpactAnalyzerEngine.
        """
        return asset_id in self

    def node_count(self) -> int:
        """Return the number of nodes in the graph.

        Alias for ``len(self)`` (delegates to ``__len__``).
        Used by ImpactAnalyzerEngine for blast radius calculation.
        """
        return len(self)

    def get_edges_between(
        self, source_id: str, target_id: str,
    ) -> List[dict]:
        """Return all edges between two specific nodes.

        Used by ImpactAnalyzerEngine for critical path analysis.

        Args:
            source_id: Source node identifier.
            target_id: Target node identifier.

        Returns:
            List of edge dictionaries connecting source to target.
        """
        with self._lock:
            edge_ids = self._adjacency_out.get(source_id, set())
            return [
                dict(self._edges[eid])
                for eid in edge_ids
                if eid in self._edges
                and self._edges[eid]["target_asset_id"] == target_id
            ]

    def get_all_nodes(self) -> List[dict]:
        """Return a list of all node dictionaries in the graph.

        Used by LineageValidatorEngine for orphan detection and
        completeness scoring.  Includes an ``id`` alias for ``asset_id``
        for compatibility with the validator's node-id resolution.

        Returns:
            List of node dictionaries (deep copies) with ``id`` key.
        """
        with self._lock:
            result = []
            for n in self._nodes.values():
                node_copy = dict(n)
                node_copy["id"] = node_copy.get("asset_id", "")
                result.append(node_copy)
            return result

    def get_all_edges(self) -> List[dict]:
        """Return a list of all edge dictionaries in the graph.

        Used by LineageValidatorEngine for broken edge detection and
        completeness scoring.  Includes ``source_id``/``target_id`` and
        ``source``/``target`` aliases for compatibility with the
        validator's edge resolution.

        Returns:
            List of edge dictionaries (deep copies) with compatibility keys.
        """
        with self._lock:
            result = []
            for e in self._edges.values():
                edge_copy = dict(e)
                edge_copy["source_id"] = edge_copy.get("source_asset_id", "")
                edge_copy["target_id"] = edge_copy.get("target_asset_id", "")
                edge_copy["source"] = edge_copy["source_id"]
                edge_copy["target"] = edge_copy["target_id"]
                result.append(edge_copy)
            return result

    def get_nodes(self) -> List[dict]:
        """Return all nodes with an ``id`` key for reporter compatibility.

        The LineageReporterEngine expects each node dict to contain an
        ``id`` key rather than ``asset_id``.  This method returns copies
        with both keys present.

        Returns:
            List of node dictionaries with ``id`` key added.
        """
        with self._lock:
            result = []
            for n in self._nodes.values():
                node_copy = dict(n)
                node_copy["id"] = node_copy.get("asset_id", "")
                result.append(node_copy)
            return result

    def get_edges(self) -> List[dict]:
        """Return all edges with ``source`` and ``target`` keys for reporter compatibility.

        The LineageReporterEngine may expect ``source`` / ``target`` keys
        in addition to ``source_asset_id`` / ``target_asset_id``.

        Returns:
            List of edge dictionaries with compatibility keys.
        """
        with self._lock:
            result = []
            for e in self._edges.values():
                edge_copy = dict(e)
                edge_copy["source"] = edge_copy.get("source_asset_id", "")
                edge_copy["target"] = edge_copy.get("target_asset_id", "")
                result.append(edge_copy)
            return result

    def list_nodes(self) -> List[dict]:
        """Alias for :meth:`get_nodes`.

        Used by LineageTrackerPipelineEngine for snapshot creation.
        """
        return self.get_nodes()

    def list_edges(self) -> List[dict]:
        """Alias for :meth:`get_edges`.

        Used by LineageTrackerPipelineEngine for snapshot creation.
        """
        return self.get_edges()

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all nodes and edges from the graph.

        Resets the graph to an empty state. Updates gauge metrics to zero.
        """
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._adjacency_out.clear()
            self._adjacency_in.clear()

        # Provenance
        self._provenance.record(
            entity_type="lineage_graph",
            entity_id="clear",
            action="graph_cleared",
        )

        # Metrics
        set_graph_node_count(0)
        set_graph_edge_count(0)

        logger.info("Lineage graph cleared")

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of nodes in the graph.

        Returns:
            Integer count of nodes.
        """
        with self._lock:
            return len(self._nodes)

    def __contains__(self, asset_id: str) -> bool:
        """Check whether a node exists in the graph.

        Args:
            asset_id: Node identifier to check.

        Returns:
            ``True`` if the node exists.
        """
        with self._lock:
            return asset_id in self._nodes

    def __repr__(self) -> str:
        """Return developer-friendly representation of the graph.

        Returns:
            String showing node and edge counts.
        """
        with self._lock:
            return (
                f"LineageGraphEngine("
                f"nodes={len(self._nodes)}, "
                f"edges={len(self._edges)})"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["LineageGraphEngine"]
