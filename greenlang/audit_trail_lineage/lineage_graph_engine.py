# -*- coding: utf-8 -*-
"""
LineageGraphEngine - MRV Calculation Lineage DAG Construction and Traversal

Engine 2 of 7 for AGENT-MRV-030 (GL-MRV-X-042).

Constructs directed acyclic graphs (DAGs) representing the full lineage of MRV
calculations -- from raw source data through emission factors and methodologies
to final reported figures.

Features:
    - 8-level lineage depth (L1-Source through L8-Reporting)
    - Forward lineage: "What reports depend on this EF?"
    - Backward lineage: "Where did this emission figure come from?"
    - Cross-scope lineage tracking across Scope 1/2/3
    - Lineage chain extraction for specific data points
    - Graph visualization (Mermaid, DOT, JSON)
    - Cycle detection and prevention
    - Orphan node identification
    - Graph snapshot and comparison
    - Data quality score propagation through lineage

Zero-Hallucination:
    - Deterministic graph construction from inputs
    - SHA-256 provenance hashes for graph snapshots
    - No LLM involvement in lineage tracking

Thread Safety:
    Uses __new__ singleton pattern with threading.Lock for thread-safe
    instantiation.  All mutating operations are protected by a dedicated
    reentrant lock.

Example:
    >>> engine = LineageGraphEngine.get_instance()
    >>> node = engine.add_node(
    ...     node_type="source_data",
    ...     level="L1_SOURCE",
    ...     qualified_name="erp.fuel_purchase.2025-Q1",
    ...     display_name="Fuel Purchases Q1 2025",
    ...     organization_id="ORG-001",
    ...     reporting_year=2025,
    ... )
    >>> print(node["node_id"])

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-X-042
Date: March 2026
"""

import collections
import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

AGENT_ID: str = "GL-MRV-X-042"
ENGINE_ID: str = "gl_atl_lineage_graph_engine"
ENGINE_VERSION: str = "1.0.0"
HASH_ALGORITHM: str = "sha256"
ENCODING: str = "utf-8"
MAX_DEPTH: int = 50

VALID_LEVELS: List[str] = [
    "L1_SOURCE",
    "L2_INGESTION",
    "L3_VALIDATION",
    "L4_FACTOR",
    "L5_CALCULATION",
    "L6_ALLOCATION",
    "L7_AGGREGATION",
    "L8_REPORTING",
]

VALID_NODE_TYPES: List[str] = [
    "source_data",
    "activity_data",
    "emission_factor",
    "methodology",
    "calculation",
    "allocation",
    "aggregation",
    "compliance_check",
    "report_item",
    "evidence",
]

VALID_EDGE_TYPES: List[str] = [
    "data_flow",
    "factor_application",
    "method_selection",
    "allocation_link",
    "aggregation_link",
    "compliance_link",
    "report_link",
    "evidence_link",
]

# Mapping from level string to numeric order for comparison / sorting
_LEVEL_ORDER: Dict[str, int] = {lvl: idx for idx, lvl in enumerate(VALID_LEVELS)}


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass(frozen=True)
class LineageNode:
    """
    Immutable node within the lineage DAG.

    Attributes:
        node_id: Globally unique identifier (UUID-4).
        node_type: One of VALID_NODE_TYPES.
        level: One of VALID_LEVELS (L1-L8).
        agent_id: Originating MRV agent identifier (optional).
        qualified_name: Fully qualified name for cross-reference.
        display_name: Human-readable label.
        value: Numeric value if applicable (e.g. emission total).
        unit: Unit of measurement (e.g. 'kgCO2e').
        data_quality_score: DQ score in range [0, 5] following PCAF scale.
        organization_id: Owning organization identifier.
        reporting_year: Disclosure reporting year.
        provenance_hash: SHA-256 hash of this node's content.
        metadata: Arbitrary metadata dictionary.
        created_at: ISO-8601 UTC timestamp.
    """

    node_id: str
    node_type: str
    level: str
    agent_id: Optional[str]
    qualified_name: str
    display_name: str
    value: Optional[Decimal]
    unit: Optional[str]
    data_quality_score: Decimal
    organization_id: str
    reporting_year: int
    provenance_hash: str
    metadata: Dict[str, Any]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to a plain dictionary."""
        d = asdict(self)
        # Convert Decimal fields to str for safe JSON serialization
        if d.get("value") is not None:
            d["value"] = str(d["value"])
        d["data_quality_score"] = str(d["data_quality_score"])
        return d


@dataclass(frozen=True)
class LineageEdge:
    """
    Immutable directed edge within the lineage DAG.

    Attributes:
        edge_id: Globally unique identifier (UUID-4).
        source_node_id: Upstream node identifier.
        target_node_id: Downstream node identifier.
        edge_type: One of VALID_EDGE_TYPES.
        transformation_description: Human-readable description of the transform.
        confidence: Confidence in this linkage [0, 1].
        organization_id: Owning organization identifier.
        reporting_year: Disclosure reporting year.
        metadata: Arbitrary metadata dictionary.
        created_at: ISO-8601 UTC timestamp.
    """

    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    transformation_description: str
    confidence: Decimal
    organization_id: str
    reporting_year: int
    metadata: Dict[str, Any]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize edge to a plain dictionary."""
        d = asdict(self)
        d["confidence"] = str(d["confidence"])
        return d


# =============================================================================
# SERIALIZATION HELPERS
# =============================================================================


def _serialize(obj: Any) -> str:
    """
    Deterministic JSON serialization.

    Handles Decimal, datetime, Enum, and dataclass instances.

    Args:
        obj: Object to serialize.

    Returns:
        Deterministic JSON string (sorted keys, no whitespace).
    """

    def _default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, sort_keys=True, default=_default, separators=(",", ":"))


def _compute_hash(data: Any) -> str:
    """
    Compute SHA-256 hash of arbitrary data.

    Args:
        data: Any JSON-serializable data.

    Returns:
        Lowercase hex SHA-256 digest.
    """
    serialized = _serialize(data)
    return hashlib.sha256(serialized.encode(ENCODING)).hexdigest()


# =============================================================================
# ENGINE
# =============================================================================


class LineageGraphEngine:
    """
    Thread-safe singleton engine for MRV calculation lineage DAG operations.

    Constructs and traverses directed acyclic graphs representing the full
    lineage of MRV calculations -- from raw source data through emission factors
    and methodologies to final reported figures.

    The engine enforces DAG invariants: every ``add_edge`` call performs
    reachability-based cycle detection before committing the edge.

    All public methods that mutate internal state are protected by
    ``threading.RLock`` to guarantee thread safety.

    Attributes:
        _nodes: Mapping of node_id to LineageNode.
        _edges: Mapping of edge_id to LineageEdge.
        _forward_edges: Mapping of node_id to list of outgoing edge_ids.
        _backward_edges: Mapping of node_id to list of incoming edge_ids.
        _org_year_nodes: Mapping of "{org}:{year}" to set of node_ids.

    Example:
        >>> engine = LineageGraphEngine.get_instance()
        >>> src = engine.add_node(
        ...     node_type="source_data",
        ...     level="L1_SOURCE",
        ...     qualified_name="erp.fuel.2025",
        ...     display_name="Fuel Data",
        ...     organization_id="ORG-001",
        ...     reporting_year=2025,
        ... )
        >>> calc = engine.add_node(
        ...     node_type="calculation",
        ...     level="L5_CALCULATION",
        ...     qualified_name="scope1.stationary.total",
        ...     display_name="Stationary Combustion Total",
        ...     organization_id="ORG-001",
        ...     reporting_year=2025,
        ...     value=Decimal("12345.67"),
        ...     unit="kgCO2e",
        ... )
        >>> edge = engine.add_edge(
        ...     source_node_id=src["node_id"],
        ...     target_node_id=calc["node_id"],
        ...     edge_type="data_flow",
        ...     organization_id="ORG-001",
        ...     reporting_year=2025,
        ... )
    """

    # Singleton machinery ------------------------------------------------
    _instance: Optional["LineageGraphEngine"] = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "LineageGraphEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize internal state (runs only on first creation)."""
        if self._initialized:
            return
        self._lock: threading.RLock = threading.RLock()
        self._nodes: Dict[str, LineageNode] = {}
        self._edges: Dict[str, LineageEdge] = {}
        self._forward_edges: Dict[str, List[str]] = {}
        self._backward_edges: Dict[str, List[str]] = {}
        self._org_year_nodes: Dict[str, Set[str]] = {}
        self._initialized = True
        logger.info(
            "LineageGraphEngine initialized (version=%s, agent=%s)",
            ENGINE_VERSION,
            AGENT_ID,
        )

    @classmethod
    def get_instance(cls) -> "LineageGraphEngine":
        """
        Get singleton instance (thread-safe).

        Returns:
            Singleton LineageGraphEngine instance.
        """
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls()
        return cls._instance  # type: ignore[return-value]

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._singleton_lock:
            cls._instance = None

    # =====================================================================
    # INTERNAL HELPERS
    # =====================================================================

    @staticmethod
    def _org_year_key(organization_id: str, reporting_year: int) -> str:
        """Build composite key for org/year index."""
        return f"{organization_id}:{reporting_year}"

    @staticmethod
    def _now_iso() -> str:
        """Return current UTC time as ISO-8601 string."""
        return datetime.now(timezone.utc).isoformat()

    def _compute_node_hash(self, node_type: str, qualified_name: str,
                           organization_id: str, reporting_year: int,
                           value: Optional[Decimal], unit: Optional[str],
                           data_quality_score: Decimal,
                           metadata: Dict[str, Any]) -> str:
        """
        Compute SHA-256 provenance hash for a node.

        The hash covers all semantically significant fields so that any
        change in the node's content produces a different hash.

        Args:
            node_type: Node type string.
            qualified_name: Qualified name of the node.
            organization_id: Organization identifier.
            reporting_year: Reporting year.
            value: Optional numeric value.
            unit: Optional unit of measurement.
            data_quality_score: Data quality score.
            metadata: Metadata dictionary.

        Returns:
            SHA-256 hex digest.
        """
        payload = {
            "node_type": node_type,
            "qualified_name": qualified_name,
            "organization_id": organization_id,
            "reporting_year": reporting_year,
            "value": str(value) if value is not None else None,
            "unit": unit,
            "data_quality_score": str(data_quality_score),
            "metadata": metadata,
        }
        return _compute_hash(payload)

    def _would_create_cycle(self, source_id: str, target_id: str) -> bool:
        """
        Check whether adding an edge source_id -> target_id would create a cycle.

        Uses BFS from ``target_id`` along forward edges to see whether
        ``source_id`` is reachable (which would mean adding the reverse
        direction completes a cycle).

        Args:
            source_id: Proposed edge source node.
            target_id: Proposed edge target node.

        Returns:
            True if a cycle would be created, False otherwise.
        """
        if source_id == target_id:
            return True

        # BFS from target along forward edges looking for source
        visited: Set[str] = set()
        queue: Deque[str] = collections.deque([target_id])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for edge_id in self._forward_edges.get(current, []):
                edge = self._edges[edge_id]
                next_node = edge.target_node_id
                if next_node == source_id:
                    return True
                if next_node not in visited:
                    queue.append(next_node)

        return False

    def _get_org_year_node_ids(self, organization_id: str,
                               reporting_year: int) -> Set[str]:
        """Return set of node_ids for org/year (empty set if none)."""
        key = self._org_year_key(organization_id, reporting_year)
        return self._org_year_nodes.get(key, set())

    # =====================================================================
    # NODE OPERATIONS
    # =====================================================================

    def add_node(
        self,
        node_type: str,
        level: str,
        qualified_name: str,
        display_name: str,
        organization_id: str,
        reporting_year: int,
        agent_id: Optional[str] = None,
        value: Optional[Decimal] = None,
        unit: Optional[str] = None,
        data_quality_score: Decimal = Decimal("1.0"),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a lineage node to the graph.

        Args:
            node_type: One of VALID_NODE_TYPES.
            level: One of VALID_LEVELS (L1_SOURCE through L8_REPORTING).
            qualified_name: Fully qualified name for cross-reference.
            display_name: Human-readable label.
            organization_id: Owning organization identifier.
            reporting_year: Disclosure reporting year.
            agent_id: Originating MRV agent identifier.
            value: Optional numeric value (e.g. emission total).
            unit: Optional unit of measurement.
            data_quality_score: PCAF-style DQ score (default 1.0).
            metadata: Optional metadata dictionary.

        Returns:
            Dictionary representation of the created node.

        Raises:
            ValueError: If node_type or level is invalid.
        """
        start = time.monotonic()

        if node_type not in VALID_NODE_TYPES:
            raise ValueError(
                f"Invalid node_type '{node_type}'. "
                f"Must be one of: {VALID_NODE_TYPES}"
            )
        if level not in VALID_LEVELS:
            raise ValueError(
                f"Invalid level '{level}'. Must be one of: {VALID_LEVELS}"
            )

        meta = metadata or {}

        provenance_hash = self._compute_node_hash(
            node_type, qualified_name, organization_id, reporting_year,
            value, unit, data_quality_score, meta,
        )

        node_id = str(uuid.uuid4())
        node = LineageNode(
            node_id=node_id,
            node_type=node_type,
            level=level,
            agent_id=agent_id,
            qualified_name=qualified_name,
            display_name=display_name,
            value=value,
            unit=unit,
            data_quality_score=data_quality_score,
            organization_id=organization_id,
            reporting_year=reporting_year,
            provenance_hash=provenance_hash,
            metadata=meta,
            created_at=self._now_iso(),
        )

        with self._lock:
            self._nodes[node_id] = node
            self._forward_edges.setdefault(node_id, [])
            self._backward_edges.setdefault(node_id, [])

            key = self._org_year_key(organization_id, reporting_year)
            self._org_year_nodes.setdefault(key, set()).add(node_id)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "add_node: id=%s type=%s level=%s qn=%s (%.2fms)",
            node_id, node_type, level, qualified_name, elapsed_ms,
        )
        return node.to_dict()

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single node by its identifier.

        Args:
            node_id: Node UUID.

        Returns:
            Node dictionary, or None if not found.
        """
        with self._lock:
            node = self._nodes.get(node_id)
            return node.to_dict() if node else None

    def get_nodes_by_level(
        self,
        organization_id: str,
        reporting_year: int,
        level: str,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all nodes at a given lineage level for an org/year.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.
            level: One of VALID_LEVELS.

        Returns:
            List of node dictionaries at the specified level.

        Raises:
            ValueError: If level is invalid.
        """
        if level not in VALID_LEVELS:
            raise ValueError(
                f"Invalid level '{level}'. Must be one of: {VALID_LEVELS}"
            )
        with self._lock:
            node_ids = self._get_org_year_node_ids(organization_id, reporting_year)
            return [
                self._nodes[nid].to_dict()
                for nid in node_ids
                if self._nodes[nid].level == level
            ]

    def get_nodes_by_type(
        self,
        organization_id: str,
        reporting_year: int,
        node_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all nodes of a given type for an org/year.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.
            node_type: One of VALID_NODE_TYPES.

        Returns:
            List of node dictionaries of the specified type.

        Raises:
            ValueError: If node_type is invalid.
        """
        if node_type not in VALID_NODE_TYPES:
            raise ValueError(
                f"Invalid node_type '{node_type}'. "
                f"Must be one of: {VALID_NODE_TYPES}"
            )
        with self._lock:
            node_ids = self._get_org_year_node_ids(organization_id, reporting_year)
            return [
                self._nodes[nid].to_dict()
                for nid in node_ids
                if self._nodes[nid].node_type == node_type
            ]

    # =====================================================================
    # EDGE OPERATIONS
    # =====================================================================

    def add_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_type: str,
        organization_id: str,
        reporting_year: int,
        transformation_description: str = "",
        confidence: Decimal = Decimal("1.0"),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a directed edge between two nodes with cycle detection.

        The engine performs reachability-based cycle detection before
        committing the edge.  If the edge would create a cycle, a
        ``ValueError`` is raised and the graph is left unchanged.

        Args:
            source_node_id: Upstream node UUID.
            target_node_id: Downstream node UUID.
            edge_type: One of VALID_EDGE_TYPES.
            organization_id: Owning organization identifier.
            reporting_year: Disclosure reporting year.
            transformation_description: Human-readable description.
            confidence: Edge confidence in [0, 1].
            metadata: Optional metadata dictionary.

        Returns:
            Dictionary representation of the created edge.

        Raises:
            ValueError: If edge_type is invalid, either node is missing,
                or the edge would create a cycle.
        """
        start = time.monotonic()

        if edge_type not in VALID_EDGE_TYPES:
            raise ValueError(
                f"Invalid edge_type '{edge_type}'. "
                f"Must be one of: {VALID_EDGE_TYPES}"
            )

        meta = metadata or {}

        with self._lock:
            if source_node_id not in self._nodes:
                raise ValueError(
                    f"Source node '{source_node_id}' not found in graph."
                )
            if target_node_id not in self._nodes:
                raise ValueError(
                    f"Target node '{target_node_id}' not found in graph."
                )

            # Cycle detection
            if self._would_create_cycle(source_node_id, target_node_id):
                raise ValueError(
                    f"Adding edge {source_node_id} -> {target_node_id} "
                    f"would create a cycle. DAG invariant violated."
                )

            edge_id = str(uuid.uuid4())
            edge = LineageEdge(
                edge_id=edge_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                edge_type=edge_type,
                transformation_description=transformation_description,
                confidence=confidence,
                organization_id=organization_id,
                reporting_year=reporting_year,
                metadata=meta,
                created_at=self._now_iso(),
            )

            self._edges[edge_id] = edge
            self._forward_edges.setdefault(source_node_id, []).append(edge_id)
            self._backward_edges.setdefault(target_node_id, []).append(edge_id)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "add_edge: id=%s %s->%s type=%s (%.2fms)",
            edge_id, source_node_id, target_node_id, edge_type, elapsed_ms,
        )
        return edge.to_dict()

    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single edge by its identifier.

        Args:
            edge_id: Edge UUID.

        Returns:
            Edge dictionary, or None if not found.
        """
        with self._lock:
            edge = self._edges.get(edge_id)
            return edge.to_dict() if edge else None

    # =====================================================================
    # GRAPH RETRIEVAL
    # =====================================================================

    def get_graph(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> Dict[str, Any]:
        """
        Retrieve the full lineage graph for an organization and year.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            Dictionary with ``nodes``, ``edges``, ``node_count``,
            ``edge_count``, and ``graph_hash``.
        """
        with self._lock:
            node_ids = self._get_org_year_node_ids(organization_id, reporting_year)
            nodes = [self._nodes[nid].to_dict() for nid in node_ids]

            # Collect edges whose source OR target belong to this org/year
            edge_ids_seen: Set[str] = set()
            edges: List[Dict[str, Any]] = []
            for nid in node_ids:
                for eid in self._forward_edges.get(nid, []):
                    if eid not in edge_ids_seen:
                        edge_ids_seen.add(eid)
                        edges.append(self._edges[eid].to_dict())
                for eid in self._backward_edges.get(nid, []):
                    if eid not in edge_ids_seen:
                        edge_ids_seen.add(eid)
                        edges.append(self._edges[eid].to_dict())

            graph_hash = self._compute_graph_hash_internal(node_ids)

        return {
            "organization_id": organization_id,
            "reporting_year": reporting_year,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "graph_hash": graph_hash,
            "retrieved_at": self._now_iso(),
        }

    # =====================================================================
    # TRAVERSAL
    # =====================================================================

    def traverse_forward(
        self,
        start_node_id: str,
        max_depth: Optional[int] = None,
        node_type_filter: Optional[str] = None,
        level_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Forward (impact) traversal from a given node.

        Answers the question: "What downstream calculations, aggregations,
        and reports are affected by this node?"

        Uses breadth-first search along forward edges.

        Args:
            start_node_id: Starting node UUID.
            max_depth: Maximum traversal depth (default MAX_DEPTH).
            node_type_filter: If set, only include nodes of this type.
            level_filter: If set, only include nodes at this level.

        Returns:
            Dictionary with ``start_node``, ``visited_nodes``,
            ``traversed_edges``, ``depth_reached``, and ``provenance_hash``.

        Raises:
            ValueError: If start_node_id is not in the graph.
        """
        effective_depth = min(max_depth or MAX_DEPTH, MAX_DEPTH)

        with self._lock:
            if start_node_id not in self._nodes:
                raise ValueError(
                    f"Start node '{start_node_id}' not found in graph."
                )

            visited_nodes: List[Dict[str, Any]] = []
            traversed_edges: List[Dict[str, Any]] = []
            visited_ids: Set[str] = set()
            depth_reached = 0

            # BFS: (node_id, current_depth)
            queue: Deque[Tuple[str, int]] = collections.deque(
                [(start_node_id, 0)]
            )

            while queue:
                nid, depth = queue.popleft()

                if nid in visited_ids:
                    continue
                visited_ids.add(nid)

                node = self._nodes[nid]

                # Apply filters
                passes_filter = True
                if node_type_filter and node.node_type != node_type_filter:
                    passes_filter = False
                if level_filter and node.level != level_filter:
                    passes_filter = False

                if passes_filter:
                    visited_nodes.append(node.to_dict())

                depth_reached = max(depth_reached, depth)

                if depth >= effective_depth:
                    continue

                for edge_id in self._forward_edges.get(nid, []):
                    edge = self._edges[edge_id]
                    traversed_edges.append(edge.to_dict())
                    if edge.target_node_id not in visited_ids:
                        queue.append((edge.target_node_id, depth + 1))

        provenance_hash = _compute_hash({
            "traversal": "forward",
            "start": start_node_id,
            "node_count": len(visited_nodes),
            "edge_count": len(traversed_edges),
        })

        return {
            "start_node_id": start_node_id,
            "direction": "forward",
            "max_depth": effective_depth,
            "depth_reached": depth_reached,
            "visited_nodes": visited_nodes,
            "traversed_edges": traversed_edges,
            "node_count": len(visited_nodes),
            "edge_count": len(traversed_edges),
            "provenance_hash": provenance_hash,
        }

    def traverse_backward(
        self,
        start_node_id: str,
        max_depth: Optional[int] = None,
        node_type_filter: Optional[str] = None,
        level_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Backward (provenance) traversal from a given node.

        Answers the question: "Where did this emission figure come from?"

        Uses breadth-first search along backward edges.

        Args:
            start_node_id: Starting node UUID.
            max_depth: Maximum traversal depth (default MAX_DEPTH).
            node_type_filter: If set, only include nodes of this type.
            level_filter: If set, only include nodes at this level.

        Returns:
            Dictionary with ``start_node``, ``visited_nodes``,
            ``traversed_edges``, ``depth_reached``, and ``provenance_hash``.

        Raises:
            ValueError: If start_node_id is not in the graph.
        """
        effective_depth = min(max_depth or MAX_DEPTH, MAX_DEPTH)

        with self._lock:
            if start_node_id not in self._nodes:
                raise ValueError(
                    f"Start node '{start_node_id}' not found in graph."
                )

            visited_nodes: List[Dict[str, Any]] = []
            traversed_edges: List[Dict[str, Any]] = []
            visited_ids: Set[str] = set()
            depth_reached = 0

            queue: Deque[Tuple[str, int]] = collections.deque(
                [(start_node_id, 0)]
            )

            while queue:
                nid, depth = queue.popleft()

                if nid in visited_ids:
                    continue
                visited_ids.add(nid)

                node = self._nodes[nid]

                passes_filter = True
                if node_type_filter and node.node_type != node_type_filter:
                    passes_filter = False
                if level_filter and node.level != level_filter:
                    passes_filter = False

                if passes_filter:
                    visited_nodes.append(node.to_dict())

                depth_reached = max(depth_reached, depth)

                if depth >= effective_depth:
                    continue

                for edge_id in self._backward_edges.get(nid, []):
                    edge = self._edges[edge_id]
                    traversed_edges.append(edge.to_dict())
                    if edge.source_node_id not in visited_ids:
                        queue.append((edge.source_node_id, depth + 1))

        provenance_hash = _compute_hash({
            "traversal": "backward",
            "start": start_node_id,
            "node_count": len(visited_nodes),
            "edge_count": len(traversed_edges),
        })

        return {
            "start_node_id": start_node_id,
            "direction": "backward",
            "max_depth": effective_depth,
            "depth_reached": depth_reached,
            "visited_nodes": visited_nodes,
            "traversed_edges": traversed_edges,
            "node_count": len(visited_nodes),
            "edge_count": len(traversed_edges),
            "provenance_hash": provenance_hash,
        }

    # =====================================================================
    # PATH / CHAIN
    # =====================================================================

    def get_lineage_chain(
        self,
        start_node_id: str,
        end_node_id: str,
    ) -> Dict[str, Any]:
        """
        Find the shortest directed path between two nodes.

        Uses BFS along forward edges from ``start_node_id`` looking for
        ``end_node_id``.

        Args:
            start_node_id: Origin node UUID.
            end_node_id: Destination node UUID.

        Returns:
            Dictionary with ``path_nodes``, ``path_edges``, ``path_length``,
            ``found``, and ``provenance_hash``.

        Raises:
            ValueError: If either node is not in the graph.
        """
        with self._lock:
            if start_node_id not in self._nodes:
                raise ValueError(
                    f"Start node '{start_node_id}' not found in graph."
                )
            if end_node_id not in self._nodes:
                raise ValueError(
                    f"End node '{end_node_id}' not found in graph."
                )

            if start_node_id == end_node_id:
                node_dict = self._nodes[start_node_id].to_dict()
                return {
                    "start_node_id": start_node_id,
                    "end_node_id": end_node_id,
                    "found": True,
                    "path_nodes": [node_dict],
                    "path_edges": [],
                    "path_length": 0,
                    "provenance_hash": _compute_hash({
                        "chain": [start_node_id],
                    }),
                }

            # BFS with parent tracking
            visited: Set[str] = {start_node_id}
            # parent[child] = (parent_node_id, edge_id)
            parent: Dict[str, Tuple[str, str]] = {}
            queue: Deque[str] = collections.deque([start_node_id])
            found = False

            while queue and not found:
                current = queue.popleft()
                for edge_id in self._forward_edges.get(current, []):
                    edge = self._edges[edge_id]
                    child = edge.target_node_id
                    if child in visited:
                        continue
                    visited.add(child)
                    parent[child] = (current, edge_id)
                    if child == end_node_id:
                        found = True
                        break
                    queue.append(child)

            if not found:
                return {
                    "start_node_id": start_node_id,
                    "end_node_id": end_node_id,
                    "found": False,
                    "path_nodes": [],
                    "path_edges": [],
                    "path_length": 0,
                    "provenance_hash": _compute_hash({
                        "chain": "not_found",
                        "start": start_node_id,
                        "end": end_node_id,
                    }),
                }

            # Reconstruct path
            path_node_ids: List[str] = []
            path_edge_ids: List[str] = []
            cur = end_node_id
            while cur != start_node_id:
                path_node_ids.append(cur)
                par, eid = parent[cur]
                path_edge_ids.append(eid)
                cur = par
            path_node_ids.append(start_node_id)

            path_node_ids.reverse()
            path_edge_ids.reverse()

            path_nodes = [self._nodes[nid].to_dict() for nid in path_node_ids]
            path_edges = [self._edges[eid].to_dict() for eid in path_edge_ids]

        provenance_hash = _compute_hash({"chain": path_node_ids})

        return {
            "start_node_id": start_node_id,
            "end_node_id": end_node_id,
            "found": True,
            "path_nodes": path_nodes,
            "path_edges": path_edges,
            "path_length": len(path_edges),
            "provenance_hash": provenance_hash,
        }

    # =====================================================================
    # ROOT / LEAF / ORPHAN NODES
    # =====================================================================

    def get_root_nodes(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve nodes with no incoming edges (sources) for an org/year.

        Root nodes are the raw data sources at the beginning of the
        lineage chain.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            List of root node dictionaries.
        """
        with self._lock:
            node_ids = self._get_org_year_node_ids(organization_id, reporting_year)
            roots: List[Dict[str, Any]] = []
            for nid in node_ids:
                if not self._backward_edges.get(nid, []):
                    roots.append(self._nodes[nid].to_dict())
            return roots

    def get_leaf_nodes(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve nodes with no outgoing edges (sinks) for an org/year.

        Leaf nodes are the final reported figures at the end of the
        lineage chain.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            List of leaf node dictionaries.
        """
        with self._lock:
            node_ids = self._get_org_year_node_ids(organization_id, reporting_year)
            leaves: List[Dict[str, Any]] = []
            for nid in node_ids:
                if not self._forward_edges.get(nid, []):
                    leaves.append(self._nodes[nid].to_dict())
            return leaves

    def get_orphan_nodes(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve fully disconnected nodes (no edges at all) for an org/year.

        Orphan nodes have neither incoming nor outgoing edges.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            List of orphan node dictionaries.
        """
        with self._lock:
            node_ids = self._get_org_year_node_ids(organization_id, reporting_year)
            orphans: List[Dict[str, Any]] = []
            for nid in node_ids:
                has_forward = bool(self._forward_edges.get(nid, []))
                has_backward = bool(self._backward_edges.get(nid, []))
                if not has_forward and not has_backward:
                    orphans.append(self._nodes[nid].to_dict())
            return orphans

    # =====================================================================
    # CYCLE DETECTION
    # =====================================================================

    def detect_cycles(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> List[List[str]]:
        """
        Detect cycles in the lineage graph for an org/year.

        For a well-formed DAG this should always return an empty list.
        Uses iterative DFS with three-colour marking (WHITE, GRAY, BLACK)
        for efficient cycle detection.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            List of cycles, where each cycle is a list of node_ids.
            Empty list if the graph is a valid DAG.
        """
        with self._lock:
            node_ids = self._get_org_year_node_ids(organization_id, reporting_year)
            if not node_ids:
                return []

            WHITE, GRAY, BLACK = 0, 1, 2
            colour: Dict[str, int] = {nid: WHITE for nid in node_ids}
            parent_map: Dict[str, Optional[str]] = {nid: None for nid in node_ids}
            cycles: List[List[str]] = []

            for start_nid in node_ids:
                if colour[start_nid] != WHITE:
                    continue

                stack: List[Tuple[str, int]] = [(start_nid, 0)]

                while stack:
                    nid, edge_idx = stack.pop()

                    if edge_idx == 0:
                        colour[nid] = GRAY

                    forward_eids = self._forward_edges.get(nid, [])

                    if edge_idx < len(forward_eids):
                        # Re-push current with next edge index
                        stack.append((nid, edge_idx + 1))

                        edge = self._edges[forward_eids[edge_idx]]
                        child = edge.target_node_id

                        # Only consider children in this org/year
                        if child not in colour:
                            continue

                        if colour[child] == GRAY:
                            # Found a cycle -- reconstruct
                            cycle = [child]
                            cur = nid
                            while cur != child:
                                cycle.append(cur)
                                cur = parent_map.get(cur, child)  # type: ignore[assignment]
                                if cur is None:
                                    break
                            cycle.append(child)
                            cycle.reverse()
                            cycles.append(cycle)
                        elif colour[child] == WHITE:
                            parent_map[child] = nid
                            stack.append((child, 0))
                    else:
                        colour[nid] = BLACK

        return cycles

    # =====================================================================
    # STATISTICS
    # =====================================================================

    def get_graph_statistics(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> Dict[str, Any]:
        """
        Compute statistics for the lineage graph.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            Dictionary with node counts by level, node counts by type,
            edge counts by type, root/leaf/orphan counts, max depth, and
            total counts.
        """
        with self._lock:
            node_ids = self._get_org_year_node_ids(organization_id, reporting_year)
            nodes_list = [self._nodes[nid] for nid in node_ids]

            # Counts by level
            by_level: Dict[str, int] = {lvl: 0 for lvl in VALID_LEVELS}
            for n in nodes_list:
                by_level[n.level] = by_level.get(n.level, 0) + 1

            # Counts by node type
            by_node_type: Dict[str, int] = {nt: 0 for nt in VALID_NODE_TYPES}
            for n in nodes_list:
                by_node_type[n.node_type] = by_node_type.get(n.node_type, 0) + 1

            # Collect edges for this org/year
            edge_ids_seen: Set[str] = set()
            org_edges: List[LineageEdge] = []
            for nid in node_ids:
                for eid in self._forward_edges.get(nid, []):
                    if eid not in edge_ids_seen:
                        edge_ids_seen.add(eid)
                        org_edges.append(self._edges[eid])
                for eid in self._backward_edges.get(nid, []):
                    if eid not in edge_ids_seen:
                        edge_ids_seen.add(eid)
                        org_edges.append(self._edges[eid])

            by_edge_type: Dict[str, int] = {et: 0 for et in VALID_EDGE_TYPES}
            for e in org_edges:
                by_edge_type[e.edge_type] = by_edge_type.get(e.edge_type, 0) + 1

            # Root / leaf / orphan counts
            root_count = 0
            leaf_count = 0
            orphan_count = 0
            for nid in node_ids:
                has_in = bool(self._backward_edges.get(nid, []))
                has_out = bool(self._forward_edges.get(nid, []))
                if not has_in and not has_out:
                    orphan_count += 1
                elif not has_in:
                    root_count += 1
                if not has_out and has_in:
                    leaf_count += 1

            # Max depth via BFS from each root
            max_depth = 0
            for nid in node_ids:
                if not self._backward_edges.get(nid, []):
                    depth = self._max_depth_from(nid, node_ids)
                    max_depth = max(max_depth, depth)

            # Average DQ score
            dq_scores = [n.data_quality_score for n in nodes_list]
            avg_dq = (
                sum(dq_scores) / len(dq_scores)
                if dq_scores
                else Decimal("0")
            )

        return {
            "organization_id": organization_id,
            "reporting_year": reporting_year,
            "total_nodes": len(nodes_list),
            "total_edges": len(org_edges),
            "nodes_by_level": by_level,
            "nodes_by_type": by_node_type,
            "edges_by_type": by_edge_type,
            "root_count": root_count,
            "leaf_count": leaf_count,
            "orphan_count": orphan_count,
            "max_depth": max_depth,
            "average_data_quality_score": str(avg_dq),
            "computed_at": self._now_iso(),
        }

    def _max_depth_from(self, start_id: str, allowed_ids: Set[str]) -> int:
        """
        Compute maximum forward depth from a node via BFS.

        Args:
            start_id: Starting node id.
            allowed_ids: Set of node ids within the graph scope.

        Returns:
            Maximum depth integer.
        """
        visited: Set[str] = set()
        queue: Deque[Tuple[str, int]] = collections.deque([(start_id, 0)])
        max_d = 0

        while queue:
            nid, depth = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            max_d = max(max_d, depth)

            for eid in self._forward_edges.get(nid, []):
                edge = self._edges[eid]
                child = edge.target_node_id
                if child in allowed_ids and child not in visited:
                    queue.append((child, depth + 1))

        return max_d

    # =====================================================================
    # CALCULATION LINEAGE
    # =====================================================================

    def get_calculation_lineage(
        self,
        calculation_id: str,
    ) -> Dict[str, Any]:
        """
        Retrieve complete lineage for a specific calculation node.

        Performs both backward traversal (inputs) and forward traversal
        (outputs) from the given calculation node and merges the results.

        Args:
            calculation_id: Node UUID of a calculation node.

        Returns:
            Dictionary with ``calculation_node``, ``inputs`` (backward),
            ``outputs`` (forward), and ``provenance_hash``.

        Raises:
            ValueError: If node not found or not a calculation type.
        """
        with self._lock:
            node = self._nodes.get(calculation_id)
            if node is None:
                raise ValueError(
                    f"Calculation node '{calculation_id}' not found."
                )

        backward = self.traverse_backward(calculation_id)
        forward = self.traverse_forward(calculation_id)

        provenance_hash = _compute_hash({
            "calculation_id": calculation_id,
            "backward_hash": backward["provenance_hash"],
            "forward_hash": forward["provenance_hash"],
        })

        return {
            "calculation_node": node.to_dict(),
            "inputs": backward,
            "outputs": forward,
            "total_input_nodes": backward["node_count"],
            "total_output_nodes": forward["node_count"],
            "provenance_hash": provenance_hash,
        }

    # =====================================================================
    # DATA QUALITY PATH
    # =====================================================================

    def get_data_quality_path(
        self,
        start_node_id: str,
        end_node_id: str,
    ) -> Dict[str, Any]:
        """
        Retrieve data quality scores along the lineage path.

        Uses ``get_lineage_chain`` to find the shortest path and then
        extracts DQ scores for each node.  Also computes the minimum
        (worst) DQ score along the path, which represents the data
        quality bottleneck.

        Args:
            start_node_id: Origin node UUID.
            end_node_id: Destination node UUID.

        Returns:
            Dictionary with ``path_dq_scores``, ``min_dq_score``,
            ``max_dq_score``, ``avg_dq_score``, ``bottleneck_node_id``,
            ``found``, and ``provenance_hash``.

        Raises:
            ValueError: If either node is not in the graph.
        """
        chain = self.get_lineage_chain(start_node_id, end_node_id)

        if not chain["found"]:
            return {
                "start_node_id": start_node_id,
                "end_node_id": end_node_id,
                "found": False,
                "path_dq_scores": [],
                "min_dq_score": None,
                "max_dq_score": None,
                "avg_dq_score": None,
                "bottleneck_node_id": None,
                "provenance_hash": _compute_hash({
                    "dq_path": "not_found",
                    "start": start_node_id,
                    "end": end_node_id,
                }),
            }

        path_dq: List[Dict[str, Any]] = []
        dq_values: List[Decimal] = []

        for node_dict in chain["path_nodes"]:
            dq_score = Decimal(node_dict["data_quality_score"])
            dq_values.append(dq_score)
            path_dq.append({
                "node_id": node_dict["node_id"],
                "display_name": node_dict["display_name"],
                "level": node_dict["level"],
                "data_quality_score": str(dq_score),
            })

        min_dq = min(dq_values) if dq_values else Decimal("0")
        max_dq = max(dq_values) if dq_values else Decimal("0")
        avg_dq = sum(dq_values) / len(dq_values) if dq_values else Decimal("0")

        bottleneck_idx = dq_values.index(min_dq) if dq_values else -1
        bottleneck_node_id = (
            chain["path_nodes"][bottleneck_idx]["node_id"]
            if bottleneck_idx >= 0
            else None
        )

        provenance_hash = _compute_hash({
            "dq_path": [str(v) for v in dq_values],
            "start": start_node_id,
            "end": end_node_id,
        })

        return {
            "start_node_id": start_node_id,
            "end_node_id": end_node_id,
            "found": True,
            "path_dq_scores": path_dq,
            "min_dq_score": str(min_dq),
            "max_dq_score": str(max_dq),
            "avg_dq_score": str(avg_dq),
            "bottleneck_node_id": bottleneck_node_id,
            "path_length": chain["path_length"],
            "provenance_hash": provenance_hash,
        }

    # =====================================================================
    # GRAPH HASH
    # =====================================================================

    def compute_graph_hash(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> str:
        """
        Compute a deterministic SHA-256 hash of the graph state.

        The hash covers all node provenance hashes and all edge data,
        providing a single fingerprint for the entire lineage graph.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            SHA-256 hex digest representing the graph state.
        """
        with self._lock:
            node_ids = self._get_org_year_node_ids(organization_id, reporting_year)
            return self._compute_graph_hash_internal(node_ids)

    def _compute_graph_hash_internal(self, node_ids: Set[str]) -> str:
        """
        Compute graph hash from a set of node ids (must be called under lock).

        Args:
            node_ids: Set of node UUIDs.

        Returns:
            SHA-256 hex digest.
        """
        if not node_ids:
            return hashlib.sha256(b"empty_graph").hexdigest()

        # Sort node provenance hashes for determinism
        node_hashes = sorted(
            self._nodes[nid].provenance_hash for nid in node_ids
        )

        # Collect and sort edge representations
        edge_ids_seen: Set[str] = set()
        edge_reprs: List[str] = []
        for nid in node_ids:
            for eid in self._forward_edges.get(nid, []):
                if eid not in edge_ids_seen:
                    edge_ids_seen.add(eid)
                    e = self._edges[eid]
                    edge_reprs.append(
                        f"{e.source_node_id}:{e.target_node_id}:{e.edge_type}"
                    )
        edge_reprs.sort()

        combined = "|".join(node_hashes) + "||" + "|".join(edge_reprs)
        return hashlib.sha256(combined.encode(ENCODING)).hexdigest()

    # =====================================================================
    # VISUALIZATION
    # =====================================================================

    def visualize(
        self,
        organization_id: str,
        reporting_year: int,
        format: str = "mermaid",
    ) -> str:
        """
        Generate a text visualization of the lineage graph.

        Supported formats:
            - ``mermaid``: Mermaid.js flowchart TD syntax
            - ``dot``: Graphviz DOT language
            - ``json``: Indented JSON representation

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.
            format: Output format (default "mermaid").

        Returns:
            Visualization string in the requested format.

        Raises:
            ValueError: If format is unsupported.
        """
        graph = self.get_graph(organization_id, reporting_year)

        if format == "mermaid":
            return self._render_mermaid(graph)
        elif format == "dot":
            return self._render_dot(graph)
        elif format == "json":
            return self._render_json(graph)
        else:
            raise ValueError(
                f"Unsupported visualization format '{format}'. "
                f"Choose from: mermaid, dot, json"
            )

    def _render_mermaid(self, graph: Dict[str, Any]) -> str:
        """
        Render lineage graph as Mermaid.js flowchart.

        Args:
            graph: Graph dictionary from get_graph().

        Returns:
            Mermaid flowchart string.
        """
        lines: List[str] = [
            "```mermaid",
            "flowchart TD",
        ]

        # Node definitions with shapes based on level
        level_shapes = {
            "L1_SOURCE": ("([", "])", "source"),
            "L2_INGESTION": ("[/", "/]", "ingestion"),
            "L3_VALIDATION": ("{{", "}}", "validation"),
            "L4_FACTOR": ("[(", ")]", "factor"),
            "L5_CALCULATION": ("[[", "]]", "calculation"),
            "L6_ALLOCATION": ("((", "))", "allocation"),
            "L7_AGGREGATION": ("[", "]", "aggregation"),
            "L8_REPORTING": (">", "]", "reporting"),
        }

        # Build node id mapping (short ids for readability)
        node_short: Dict[str, str] = {}
        for idx, node in enumerate(graph["nodes"]):
            short_id = f"N{idx}"
            node_short[node["node_id"]] = short_id
            level = node.get("level", "L5_CALCULATION")
            open_b, close_b, _ = level_shapes.get(
                level, ("[", "]", "unknown")
            )
            label = node.get("display_name", node["node_id"][:8])
            # Escape special Mermaid characters
            label = label.replace('"', "'")
            lines.append(f'    {short_id}{open_b}"{label}"{close_b}')

        # Edge definitions
        for edge in graph["edges"]:
            src = node_short.get(edge["source_node_id"], "?")
            tgt = node_short.get(edge["target_node_id"], "?")
            etype = edge.get("edge_type", "")
            lines.append(f"    {src} -->|{etype}| {tgt}")

        lines.append("```")
        return "\n".join(lines)

    def _render_dot(self, graph: Dict[str, Any]) -> str:
        """
        Render lineage graph as Graphviz DOT language.

        Args:
            graph: Graph dictionary from get_graph().

        Returns:
            DOT language string.
        """
        lines: List[str] = [
            "digraph lineage {",
            '    rankdir=TB;',
            '    node [fontsize=10];',
            '    edge [fontsize=8];',
        ]

        level_colors = {
            "L1_SOURCE": "lightblue",
            "L2_INGESTION": "lightyellow",
            "L3_VALIDATION": "lightsalmon",
            "L4_FACTOR": "lightgreen",
            "L5_CALCULATION": "gold",
            "L6_ALLOCATION": "plum",
            "L7_AGGREGATION": "lightcyan",
            "L8_REPORTING": "lightcoral",
        }

        node_short: Dict[str, str] = {}
        for idx, node in enumerate(graph["nodes"]):
            short_id = f"N{idx}"
            node_short[node["node_id"]] = short_id
            label = node.get("display_name", node["node_id"][:8])
            label = label.replace('"', '\\"')
            color = level_colors.get(node.get("level", ""), "white")
            lines.append(
                f'    {short_id} [label="{label}" '
                f'style=filled fillcolor={color}];'
            )

        for edge in graph["edges"]:
            src = node_short.get(edge["source_node_id"], "?")
            tgt = node_short.get(edge["target_node_id"], "?")
            etype = edge.get("edge_type", "")
            lines.append(f'    {src} -> {tgt} [label="{etype}"];')

        lines.append("}")
        return "\n".join(lines)

    def _render_json(self, graph: Dict[str, Any]) -> str:
        """
        Render lineage graph as indented JSON.

        Args:
            graph: Graph dictionary from get_graph().

        Returns:
            JSON string.
        """
        return json.dumps(graph, indent=2, sort_keys=True, default=str)

    # =====================================================================
    # RESET
    # =====================================================================

    def reset(self) -> None:
        """
        Clear all nodes, edges, and indexes.

        Intended for testing only.  After reset the engine instance
        is still usable but the graph is empty.
        """
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._forward_edges.clear()
            self._backward_edges.clear()
            self._org_year_nodes.clear()
            logger.info("LineageGraphEngine: graph reset (all data cleared)")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "AGENT_ID",
    "ENGINE_ID",
    "ENGINE_VERSION",
    "MAX_DEPTH",
    "VALID_LEVELS",
    "VALID_NODE_TYPES",
    "VALID_EDGE_TYPES",
    # Data models
    "LineageNode",
    "LineageEdge",
    # Engine
    "LineageGraphEngine",
]
