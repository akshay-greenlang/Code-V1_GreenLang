# -*- coding: utf-8 -*-
"""
SupplyChainGraphEngine - AGENT-EUDR-001 Feature 1: Supply Chain Graph Engine

This module implements the core graph engine for modeling EUDR supply chains
as directed acyclic graphs (DAGs). It provides typed nodes representing supply
chain actors (Producer, Collector, Processor, Trader, Importer, etc.) and
directed edges representing custody transfers with commodity, quantity, date,
and batch metadata.

Key capabilities:
- DAG-based graph with typed nodes and directed edges
- Node/edge CRUD operations with full audit trail
- Cycle detection to enforce acyclic EUDR supply chain topology
- Topological sorting for processing order determination
- Graph serialization to/from JSON, GraphML, and internal binary format
- NetworkX for in-memory graph operations
- PostgreSQL persistence layer with psycopg async
- Graph versioning with immutable snapshots
- Support for 100,000+ nodes with <1ms single-node lookup
- SHA-256 provenance hashes on all graph mutations

Reference: Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR)
PRD: PRD-AGENT-EUDR-001, Feature 1
DB Schema: V089 (eudr_supply_chain_mapper schema)

Example:
    >>> engine = SupplyChainGraphEngine(config)
    >>> await engine.initialize()
    >>>
    >>> graph_id = await engine.create_graph(
    ...     operator_id="op-123",
    ...     commodity="cocoa",
    ...     graph_name="Ghana Cocoa Supply Chain"
    ... )
    >>>
    >>> node_id = await engine.add_node(
    ...     graph_id=graph_id,
    ...     node_type=NodeType.PRODUCER,
    ...     operator_name="Cooperative Alpha",
    ...     country_code="GH",
    ... )
    >>>
    >>> edge_id = await engine.add_edge(
    ...     graph_id=graph_id,
    ...     source_node_id=producer_id,
    ...     target_node_id=collector_id,
    ...     commodity="cocoa",
    ...     quantity=Decimal("5000"),
    ... )

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001 Supply Chain Mapping Master
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import pickle
import struct
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from uuid import uuid4

from pydantic import Field, field_validator
from greenlang.schemas import GreenLangBase
from greenlang.utilities.exceptions.compliance import ComplianceException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None  # type: ignore[assignment]
    NETWORKX_AVAILABLE = False

try:
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    PSYCOPG_AVAILABLE = True
except ImportError:
    AsyncConnection = None  # type: ignore[assignment, misc]
    AsyncConnectionPool = None  # type: ignore[assignment, misc]
    dict_row = None  # type: ignore[assignment]
    PSYCOPG_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional canonical serialization
# ---------------------------------------------------------------------------

try:
    from greenlang.utilities.serialization.canonical import canonical_dumps
except ImportError:

    def canonical_dumps(obj: Any) -> str:
        """Fallback canonical JSON serialization."""
        return json.dumps(
            obj, sort_keys=True, separators=(",", ":"), default=str
        )


# ===================================================================
# Constants
# ===================================================================

SCHEMA = "eudr_supply_chain_mapper"
GENESIS_HASH = "genesis"
BINARY_MAGIC = b"GLSC"
BINARY_VERSION = 1


# ===================================================================
# Enums
# ===================================================================


class NodeType(str, Enum):
    """Supply chain actor types per EUDR Article 2.

    Each node in the supply chain graph represents an actor with a
    defined role in the custody chain from production plot to EU market.
    """

    PRODUCER = "producer"
    COLLECTOR = "collector"
    PROCESSOR = "processor"
    TRADER = "trader"
    IMPORTER = "importer"
    CERTIFIER = "certifier"
    WAREHOUSE = "warehouse"
    PORT = "port"


class CustodyModel(str, Enum):
    """Chain of custody models per EUDR requirements.

    Defines how physical commodity flows are tracked through
    processing and transformation steps.
    """

    IDENTITY_PRESERVED = "identity_preserved"
    SEGREGATED = "segregated"
    MASS_BALANCE = "mass_balance"


class RiskLevel(str, Enum):
    """Risk classification per EUDR Article 29 benchmarking."""

    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


class ComplianceStatus(str, Enum):
    """Compliance status for supply chain actors and edges."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_VERIFICATION = "pending_verification"
    UNDER_REVIEW = "under_review"
    INSUFFICIENT_DATA = "insufficient_data"
    EXEMPTED = "exempted"


class MutationType(str, Enum):
    """Types of graph mutations recorded in the audit trail."""

    NODE_ADDED = "node_added"
    NODE_REMOVED = "node_removed"
    NODE_UPDATED = "node_updated"
    EDGE_ADDED = "edge_added"
    EDGE_REMOVED = "edge_removed"
    EDGE_UPDATED = "edge_updated"
    GRAPH_CREATED = "graph_created"
    GRAPH_SNAPSHOT = "graph_snapshot"


# ===================================================================
# Data Models (Pydantic v2)
# ===================================================================


class SupplyChainNode(GreenLangBase):
    """A typed node in the EUDR supply chain graph.

    Represents a single actor (producer, collector, processor, etc.)
    in the supply chain, with all attributes required for EUDR
    compliance tracking.

    Attributes:
        node_id: Unique node identifier (UUID).
        node_type: Actor role in supply chain.
        operator_id: External operator/company identifier.
        operator_name: Human-readable operator name.
        country_code: ISO 3166-1 alpha-2 country code.
        region: Sub-national administrative region.
        latitude: WGS84 latitude (6+ decimal places).
        longitude: WGS84 longitude (6+ decimal places).
        commodities: EUDR commodities handled by this actor.
        tier_depth: Distance from importer (0 = importer).
        risk_score: Composite risk score (0-100).
        risk_level: LOW/STANDARD/HIGH per Article 29.
        compliance_status: Current compliance status.
        certifications: Certification schemes (FSC, RSPO, etc.).
        plot_ids: Linked production plot IDs (producers only).
        metadata: Arbitrary additional attributes.
        created_at: Node creation timestamp.
        updated_at: Last modification timestamp.
    """

    node_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique node identifier",
    )
    node_type: NodeType = Field(
        ..., description="Actor role in supply chain"
    )
    operator_id: Optional[str] = Field(
        None, description="External operator/company identifier"
    )
    operator_name: str = Field(
        ..., description="Human-readable operator name"
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    region: Optional[str] = Field(
        None, description="Sub-national region"
    )
    latitude: Optional[float] = Field(
        None, ge=-90.0, le=90.0, description="WGS84 latitude"
    )
    longitude: Optional[float] = Field(
        None, ge=-180.0, le=180.0, description="WGS84 longitude"
    )
    commodities: List[str] = Field(
        default_factory=list,
        description="EUDR commodities handled",
    )
    tier_depth: int = Field(
        default=0, ge=0, description="Distance from importer"
    )
    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Composite risk score",
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.STANDARD,
        description="Risk classification",
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.PENDING_VERIFICATION,
        description="Compliance status",
    )
    certifications: List[str] = Field(
        default_factory=list,
        description="Certification schemes",
    )
    plot_ids: List[str] = Field(
        default_factory=list,
        description="Linked production plot IDs",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional attributes",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last modification timestamp",
    )

    @field_validator("country_code")
    @classmethod
    def uppercase_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper()

    def to_db_dict(self) -> Dict[str, Any]:
        """Serialize for database insertion."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "country_code": self.country_code,
            "region": self.region,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "commodities": json.dumps(self.commodities),
            "tier_depth": self.tier_depth,
            "risk_score": float(self.risk_score),
            "risk_level": self.risk_level.value,
            "compliance_status": self.compliance_status.value,
            "certifications": json.dumps(self.certifications),
            "plot_ids": json.dumps(self.plot_ids),
            "metadata": json.dumps(self.metadata, default=str),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class SupplyChainEdge(GreenLangBase):
    """A directed edge representing a custody transfer.

    Models the transfer of an EUDR commodity from one supply chain
    actor to another, with complete traceability metadata required
    for Due Diligence Statement generation.

    Attributes:
        edge_id: Unique edge identifier (UUID).
        source_node_id: Upstream node (sender).
        target_node_id: Downstream node (receiver).
        commodity: EUDR commodity being transferred.
        product_description: Human-readable product description.
        quantity: Transfer quantity.
        unit: Unit of measure (default: kg).
        batch_number: Batch/lot identifier.
        custody_model: Chain of custody model.
        transfer_date: Date/time of custody transfer.
        cn_code: Combined Nomenclature code (EU).
        hs_code: Harmonized System code.
        transport_mode: Mode of transport.
        provenance_hash: SHA-256 hash for audit integrity.
        created_at: Edge creation timestamp.
    """

    edge_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique edge identifier",
    )
    source_node_id: str = Field(
        ..., description="Upstream node ID"
    )
    target_node_id: str = Field(
        ..., description="Downstream node ID"
    )
    commodity: str = Field(
        ..., description="EUDR commodity"
    )
    product_description: Optional[str] = Field(
        None, description="Product description"
    )
    quantity: Decimal = Field(
        ..., gt=0, description="Transfer quantity"
    )
    unit: str = Field(
        default="kg", description="Unit of measure"
    )
    batch_number: Optional[str] = Field(
        None, description="Batch/lot identifier"
    )
    custody_model: CustodyModel = Field(
        default=CustodyModel.SEGREGATED,
        description="Chain of custody model",
    )
    transfer_date: Optional[datetime] = Field(
        None, description="Transfer date/time"
    )
    cn_code: Optional[str] = Field(
        None, description="Combined Nomenclature code"
    )
    hs_code: Optional[str] = Field(
        None, description="Harmonized System code"
    )
    transport_mode: Optional[str] = Field(
        None, description="Mode of transport"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 provenance hash for this edge.

        Returns:
            SHA-256 hex digest of deterministic edge content.
        """
        hash_data = canonical_dumps(
            {
                "edge_id": self.edge_id,
                "source": self.source_node_id,
                "target": self.target_node_id,
                "commodity": self.commodity,
                "quantity": str(self.quantity),
                "batch": self.batch_number,
                "date": (
                    self.transfer_date.isoformat()
                    if self.transfer_date
                    else None
                ),
            }
        )
        return hashlib.sha256(hash_data.encode()).hexdigest()

    def to_db_dict(self) -> Dict[str, Any]:
        """Serialize for database insertion."""
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "commodity": self.commodity,
            "product_description": self.product_description,
            "quantity": float(self.quantity),
            "unit": self.unit,
            "batch_number": self.batch_number,
            "custody_model": self.custody_model.value,
            "transfer_date": self.transfer_date,
            "cn_code": self.cn_code,
            "hs_code": self.hs_code,
            "transport_mode": self.transport_mode,
            "provenance_hash": self.provenance_hash,
            "created_at": self.created_at,
        }


class GraphMutationRecord(GreenLangBase):
    """Audit trail record for a single graph mutation.

    Every add, remove, or update operation on the graph is recorded
    as an immutable mutation record for EUDR Article 31 compliance
    (5-year record keeping requirement).

    Attributes:
        mutation_id: Unique mutation identifier.
        graph_id: Graph that was mutated.
        mutation_type: Type of mutation.
        target_id: ID of the affected node or edge.
        timestamp: UTC timestamp of the mutation.
        actor: Identity of the user/system that performed it.
        previous_state: State before mutation (for updates/deletes).
        new_state: State after mutation (for adds/updates).
        provenance_hash: SHA-256 hash linking to prior mutation.
    """

    mutation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique mutation identifier",
    )
    graph_id: str = Field(..., description="Graph identifier")
    mutation_type: MutationType = Field(
        ..., description="Type of mutation"
    )
    target_id: str = Field(
        ..., description="Affected node or edge ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Mutation timestamp",
    )
    actor: str = Field(
        default="system", description="Actor identity"
    )
    previous_state: Optional[Dict[str, Any]] = Field(
        None, description="State before mutation"
    )
    new_state: Optional[Dict[str, Any]] = Field(
        None, description="State after mutation"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    def calculate_hash(self, prev_hash: str = GENESIS_HASH) -> str:
        """Calculate SHA-256 hash chaining to previous mutation.

        Args:
            prev_hash: Hash of the previous mutation record.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = canonical_dumps(
            {
                "mutation_id": self.mutation_id,
                "graph_id": self.graph_id,
                "type": self.mutation_type.value,
                "target": self.target_id,
                "timestamp": self.timestamp.isoformat(),
                "actor": self.actor,
                "prev_hash": prev_hash,
            }
        )
        return hashlib.sha256(hash_data.encode()).hexdigest()


class GraphSnapshot(GreenLangBase):
    """Immutable point-in-time snapshot of a supply chain graph.

    Captures the complete graph state for audit trail and versioning
    per EUDR Article 31 (5-year record keeping). Snapshots are
    immutable once created.

    Attributes:
        snapshot_id: Unique snapshot identifier.
        graph_id: Source graph identifier.
        version: Graph version at snapshot time.
        node_count: Number of nodes in snapshot.
        edge_count: Number of edges in snapshot.
        nodes: Complete node data.
        edges: Complete edge data.
        provenance_hash: SHA-256 hash of snapshot content.
        created_by: Identity of snapshot creator.
        created_at: Snapshot creation timestamp.
    """

    snapshot_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique snapshot identifier",
    )
    graph_id: str = Field(..., description="Source graph identifier")
    version: int = Field(..., description="Graph version")
    node_count: int = Field(default=0, description="Number of nodes")
    edge_count: int = Field(default=0, description="Number of edges")
    nodes: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Node data"
    )
    edges: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Edge data"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 content hash"
    )
    created_by: str = Field(
        default="system", description="Snapshot creator"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of the snapshot content.

        Returns:
            SHA-256 hex digest of all nodes and edges.
        """
        hash_data = canonical_dumps(
            {
                "graph_id": self.graph_id,
                "version": self.version,
                "nodes": self.nodes,
                "edges": self.edges,
            }
        )
        return hashlib.sha256(hash_data.encode()).hexdigest()


# ===================================================================
# Configuration
# ===================================================================


class GraphEngineConfig(GreenLangBase):
    """Configuration for SupplyChainGraphEngine.

    Attributes:
        database_url: PostgreSQL connection string.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        enable_persistence: Whether to persist to PostgreSQL.
        enable_audit_trail: Whether to record mutation audit trail.
        enable_snapshots: Whether to auto-create snapshots on version.
        snapshot_interval: Create snapshot every N mutations.
        max_graph_nodes: Maximum nodes per graph (memory safety).
    """

    database_url: str = Field(
        default="",
        description="PostgreSQL connection string",
    )
    pool_min_size: int = Field(default=2, ge=1, description="Min pool")
    pool_max_size: int = Field(default=10, ge=1, description="Max pool")
    enable_persistence: bool = Field(
        default=True, description="Persist to PostgreSQL"
    )
    enable_audit_trail: bool = Field(
        default=True, description="Record mutation audit trail"
    )
    enable_snapshots: bool = Field(
        default=True, description="Auto-create snapshots"
    )
    snapshot_interval: int = Field(
        default=100, ge=1, description="Snapshot every N mutations"
    )
    max_graph_nodes: int = Field(
        default=500_000,
        ge=1,
        description="Maximum nodes per graph",
    )


# ===================================================================
# Exceptions
# ===================================================================


class GraphEngineError(ComplianceException):
    """Base exception for graph engine errors."""


class CycleDetectedError(GraphEngineError):
    """Raised when adding an edge would create a cycle."""

    def __init__(
        self,
        message: str,
        source_node_id: str = "",
        target_node_id: str = "",
        cycle_path: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.cycle_path = cycle_path or []


class NodeNotFoundError(GraphEngineError):
    """Raised when a referenced node does not exist."""


class EdgeNotFoundError(GraphEngineError):
    """Raised when a referenced edge does not exist."""


class GraphNotFoundError(GraphEngineError):
    """Raised when a referenced graph does not exist."""


class GraphCapacityError(GraphEngineError):
    """Raised when a graph exceeds its maximum node capacity."""


class PersistenceError(GraphEngineError):
    """Raised when a database operation fails."""


# ===================================================================
# SupplyChainGraphEngine
# ===================================================================


class SupplyChainGraphEngine:
    """Core graph engine for EUDR supply chain modeling.

    Manages in-memory NetworkX directed graphs backed by PostgreSQL
    persistence. Provides typed node/edge CRUD, cycle detection,
    topological sorting, serialization, versioning, and full audit
    trail of all graph mutations.

    The engine uses NetworkX DiGraph for in-memory graph operations,
    providing O(1) node/edge lookups and efficient graph algorithms.
    All mutations are recorded as hash-chained audit records.

    Architecture:
        - In-memory layer: NetworkX DiGraph per loaded graph
        - Persistence layer: PostgreSQL + TimescaleDB (V089 schema)
        - Audit layer: Append-only mutation log with SHA-256 chains
        - Versioning: Immutable JSON snapshots on configurable interval

    Performance targets:
        - Single-node lookup: <1ms (dict-based, O(1))
        - Graph construction: <5s for 10,000 nodes
        - Cycle detection: <100ms for 10,000 nodes
        - Topological sort: <100ms for 10,000 nodes

    Example:
        >>> config = GraphEngineConfig(database_url="postgresql://...")
        >>> engine = SupplyChainGraphEngine(config)
        >>> await engine.initialize()
        >>>
        >>> graph_id = await engine.create_graph("op-1", "cocoa")
        >>> node_id = await engine.add_node(graph_id, NodeType.PRODUCER,
        ...     operator_name="Farm A", country_code="GH")
        >>> sorted_nodes = engine.topological_sort(graph_id)
    """

    def __init__(self, config: Optional[GraphEngineConfig] = None) -> None:
        """Initialize SupplyChainGraphEngine.

        Args:
            config: Engine configuration. Uses defaults if None.
        """
        self._config = config or GraphEngineConfig()

        # In-memory graph storage: graph_id -> NetworkX DiGraph
        self._graphs: Dict[str, "nx.DiGraph"] = {}

        # Node/edge model storage for fast attribute access
        self._nodes: Dict[str, Dict[str, SupplyChainNode]] = {}
        self._edges: Dict[str, Dict[str, SupplyChainEdge]] = {}

        # Graph metadata
        self._graph_meta: Dict[str, Dict[str, Any]] = {}

        # Audit trail: graph_id -> list of mutation records
        self._audit_trail: Dict[str, List[GraphMutationRecord]] = {}
        self._latest_mutation_hash: Dict[str, str] = {}

        # Version tracking
        self._graph_versions: Dict[str, int] = {}
        self._mutation_counters: Dict[str, int] = {}

        # Persistence pool (initialized lazily)
        self._pool: Optional[Any] = None
        self._initialized = False

        logger.info(
            "SupplyChainGraphEngine created: persistence=%s, "
            "audit=%s, snapshots=%s",
            self._config.enable_persistence,
            self._config.enable_audit_trail,
            self._config.enable_snapshots,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the engine and database connection pool.

        Creates the psycopg async connection pool if persistence is
        enabled. Must be called before any database operations.

        Raises:
            ImportError: If psycopg is not installed.
            PersistenceError: If database connection fails.
        """
        if self._config.enable_persistence:
            if not PSYCOPG_AVAILABLE:
                raise ImportError(
                    "psycopg[binary] and psycopg_pool are required for "
                    "persistence. Install with: "
                    "pip install 'psycopg[binary]' psycopg_pool"
                )
            if not self._config.database_url:
                raise PersistenceError(
                    "database_url is required when persistence is enabled"
                )
            try:
                self._pool = AsyncConnectionPool(
                    conninfo=self._config.database_url,
                    min_size=self._config.pool_min_size,
                    max_size=self._config.pool_max_size,
                    kwargs={"row_factory": dict_row},
                )
                await self._pool.open()
                logger.info("Database connection pool opened")
            except Exception as exc:
                raise PersistenceError(
                    f"Failed to open database pool: {exc}"
                ) from exc

        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "networkx is required for graph operations. "
                "Install with: pip install networkx"
            )

        self._initialized = True
        logger.info("SupplyChainGraphEngine initialized")

    async def close(self) -> None:
        """Close the engine and release database connections."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection pool closed")
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Verify engine is initialized before operations."""
        if not self._initialized:
            raise GraphEngineError(
                "Engine not initialized. Call initialize() first."
            )

    # ------------------------------------------------------------------
    # Graph lifecycle
    # ------------------------------------------------------------------

    async def create_graph(
        self,
        operator_id: str,
        commodity: str,
        graph_name: Optional[str] = None,
        graph_id: Optional[str] = None,
        actor: str = "system",
    ) -> str:
        """Create a new supply chain graph.

        Args:
            operator_id: Owner operator identifier.
            commodity: Primary EUDR commodity.
            graph_name: Human-readable graph name.
            graph_id: Optional explicit graph ID.
            actor: Identity performing the operation.

        Returns:
            Created graph identifier.

        Raises:
            GraphEngineError: If engine is not initialized.
        """
        self._ensure_initialized()
        start = time.monotonic()

        gid = graph_id or str(uuid4())
        now = datetime.now(timezone.utc)

        # Create NetworkX directed graph
        self._graphs[gid] = nx.DiGraph()
        self._nodes[gid] = {}
        self._edges[gid] = {}
        self._graph_versions[gid] = 1
        self._mutation_counters[gid] = 0
        self._audit_trail[gid] = []
        self._latest_mutation_hash[gid] = GENESIS_HASH

        self._graph_meta[gid] = {
            "graph_id": gid,
            "operator_id": operator_id,
            "commodity": commodity,
            "graph_name": graph_name or f"{commodity} supply chain",
            "total_nodes": 0,
            "total_edges": 0,
            "max_tier_depth": 0,
            "traceability_score": 0.0,
            "compliance_readiness": 0.0,
            "risk_summary": {},
            "version": 1,
            "created_at": now,
            "updated_at": now,
        }

        # Record audit
        self._record_mutation(
            graph_id=gid,
            mutation_type=MutationType.GRAPH_CREATED,
            target_id=gid,
            actor=actor,
            new_state={"operator_id": operator_id, "commodity": commodity},
        )

        # Persist
        if self._config.enable_persistence and self._pool:
            await self._persist_graph_create(gid)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Created graph %s for operator=%s commodity=%s (%.1fms)",
            gid,
            operator_id,
            commodity,
            elapsed_ms,
        )
        return gid

    async def delete_graph(self, graph_id: str) -> None:
        """Delete a graph and all its nodes, edges, and audit records.

        Args:
            graph_id: Graph to delete.

        Raises:
            GraphNotFoundError: If graph does not exist.
        """
        self._ensure_initialized()
        self._ensure_graph_exists(graph_id)

        # Remove from memory
        self._graphs.pop(graph_id, None)
        self._nodes.pop(graph_id, None)
        self._edges.pop(graph_id, None)
        self._graph_meta.pop(graph_id, None)
        self._graph_versions.pop(graph_id, None)
        self._mutation_counters.pop(graph_id, None)
        self._audit_trail.pop(graph_id, None)
        self._latest_mutation_hash.pop(graph_id, None)

        # Persist deletion
        if self._config.enable_persistence and self._pool:
            await self._persist_graph_delete(graph_id)

        logger.info("Deleted graph %s", graph_id)

    def get_graph_metadata(self, graph_id: str) -> Dict[str, Any]:
        """Get metadata for a graph.

        Args:
            graph_id: Graph identifier.

        Returns:
            Graph metadata dictionary.

        Raises:
            GraphNotFoundError: If graph does not exist.
        """
        self._ensure_graph_exists(graph_id)
        meta = dict(self._graph_meta[graph_id])
        meta["total_nodes"] = len(self._nodes.get(graph_id, {}))
        meta["total_edges"] = len(self._edges.get(graph_id, {}))
        return meta

    def list_graphs(self) -> List[Dict[str, Any]]:
        """List all loaded graphs with summary metadata.

        Returns:
            List of graph metadata dictionaries.
        """
        results = []
        for gid in self._graph_meta:
            results.append(self.get_graph_metadata(gid))
        return results

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    async def add_node(
        self,
        graph_id: str,
        node_type: NodeType,
        operator_name: str,
        country_code: str,
        node_id: Optional[str] = None,
        actor: str = "system",
        **kwargs: Any,
    ) -> str:
        """Add a typed node to the supply chain graph.

        Args:
            graph_id: Target graph identifier.
            node_type: Actor role (producer, collector, etc.).
            operator_name: Human-readable name.
            country_code: ISO 3166-1 alpha-2 country code.
            node_id: Optional explicit node ID.
            actor: Identity performing the operation.
            **kwargs: Additional SupplyChainNode attributes.

        Returns:
            Created node identifier.

        Raises:
            GraphNotFoundError: If graph does not exist.
            GraphCapacityError: If graph exceeds max nodes.
            ValueError: If node_id already exists.
        """
        self._ensure_initialized()
        self._ensure_graph_exists(graph_id)
        start = time.monotonic()

        # Capacity check
        current_count = len(self._nodes[graph_id])
        if current_count >= self._config.max_graph_nodes:
            raise GraphCapacityError(
                f"Graph {graph_id} has reached maximum capacity "
                f"of {self._config.max_graph_nodes} nodes"
            )

        nid = node_id or str(uuid4())

        if nid in self._nodes[graph_id]:
            raise ValueError(
                f"Node '{nid}' already exists in graph {graph_id}"
            )

        node = SupplyChainNode(
            node_id=nid,
            node_type=node_type,
            operator_name=operator_name,
            country_code=country_code,
            **kwargs,
        )

        # Add to NetworkX graph
        self._graphs[graph_id].add_node(
            nid,
            node_type=node_type.value,
            operator_name=operator_name,
            country_code=country_code,
        )

        # Store model
        self._nodes[graph_id][nid] = node

        # Update metadata
        self._increment_version(graph_id)

        # Record audit
        self._record_mutation(
            graph_id=graph_id,
            mutation_type=MutationType.NODE_ADDED,
            target_id=nid,
            actor=actor,
            new_state=node.model_dump(mode="json"),
        )

        # Auto-snapshot check
        await self._maybe_auto_snapshot(graph_id, actor)

        # Persist
        if self._config.enable_persistence and self._pool:
            await self._persist_node_add(graph_id, node)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "Added node %s (%s) to graph %s (%.1fms)",
            nid,
            node_type.value,
            graph_id,
            elapsed_ms,
        )
        return nid

    def get_node(
        self, graph_id: str, node_id: str
    ) -> SupplyChainNode:
        """Get a node by its identifier.

        Args:
            graph_id: Graph identifier.
            node_id: Node identifier.

        Returns:
            SupplyChainNode instance.

        Raises:
            GraphNotFoundError: If graph does not exist.
            NodeNotFoundError: If node does not exist.
        """
        self._ensure_graph_exists(graph_id)
        node = self._nodes[graph_id].get(node_id)
        if node is None:
            raise NodeNotFoundError(
                f"Node '{node_id}' not found in graph {graph_id}"
            )
        return node

    def list_nodes(
        self,
        graph_id: str,
        node_type: Optional[NodeType] = None,
        country_code: Optional[str] = None,
        risk_level: Optional[RiskLevel] = None,
    ) -> List[SupplyChainNode]:
        """List nodes with optional filters.

        Args:
            graph_id: Graph identifier.
            node_type: Filter by node type.
            country_code: Filter by country code.
            risk_level: Filter by risk level.

        Returns:
            List of matching nodes.

        Raises:
            GraphNotFoundError: If graph does not exist.
        """
        self._ensure_graph_exists(graph_id)
        nodes = list(self._nodes[graph_id].values())

        if node_type is not None:
            nodes = [n for n in nodes if n.node_type == node_type]
        if country_code is not None:
            code = country_code.upper()
            nodes = [n for n in nodes if n.country_code == code]
        if risk_level is not None:
            nodes = [n for n in nodes if n.risk_level == risk_level]

        return nodes

    async def update_node_attributes(
        self,
        graph_id: str,
        node_id: str,
        actor: str = "system",
        **attributes: Any,
    ) -> SupplyChainNode:
        """Update attributes of an existing node.

        Only the provided attributes are updated; unspecified attributes
        remain unchanged.

        Args:
            graph_id: Graph identifier.
            node_id: Node to update.
            actor: Identity performing the operation.
            **attributes: Attribute key-value pairs to update.

        Returns:
            Updated SupplyChainNode.

        Raises:
            GraphNotFoundError: If graph does not exist.
            NodeNotFoundError: If node does not exist.
        """
        self._ensure_initialized()
        self._ensure_graph_exists(graph_id)

        node = self._nodes[graph_id].get(node_id)
        if node is None:
            raise NodeNotFoundError(
                f"Node '{node_id}' not found in graph {graph_id}"
            )

        previous_state = node.model_dump(mode="json")

        # Apply updates
        for key, value in attributes.items():
            if hasattr(node, key) and key not in ("node_id", "created_at"):
                setattr(node, key, value)

        node.updated_at = datetime.now(timezone.utc)

        # Update NetworkX node data
        nx_data = self._graphs[graph_id].nodes[node_id]
        for key, value in attributes.items():
            if isinstance(value, Enum):
                nx_data[key] = value.value
            else:
                nx_data[key] = value

        # Version and audit
        self._increment_version(graph_id)
        self._record_mutation(
            graph_id=graph_id,
            mutation_type=MutationType.NODE_UPDATED,
            target_id=node_id,
            actor=actor,
            previous_state=previous_state,
            new_state=node.model_dump(mode="json"),
        )

        # Persist
        if self._config.enable_persistence and self._pool:
            await self._persist_node_update(graph_id, node)

        logger.debug("Updated node %s in graph %s", node_id, graph_id)
        return node

    async def remove_node(
        self,
        graph_id: str,
        node_id: str,
        actor: str = "system",
    ) -> None:
        """Remove a node and all its incident edges from the graph.

        Args:
            graph_id: Graph identifier.
            node_id: Node to remove.
            actor: Identity performing the operation.

        Raises:
            GraphNotFoundError: If graph does not exist.
            NodeNotFoundError: If node does not exist.
        """
        self._ensure_initialized()
        self._ensure_graph_exists(graph_id)

        node = self._nodes[graph_id].get(node_id)
        if node is None:
            raise NodeNotFoundError(
                f"Node '{node_id}' not found in graph {graph_id}"
            )

        previous_state = node.model_dump(mode="json")

        # Remove incident edges first
        edges_to_remove = [
            eid
            for eid, edge in self._edges[graph_id].items()
            if edge.source_node_id == node_id
            or edge.target_node_id == node_id
        ]
        for eid in edges_to_remove:
            edge = self._edges[graph_id].pop(eid)
            self._record_mutation(
                graph_id=graph_id,
                mutation_type=MutationType.EDGE_REMOVED,
                target_id=eid,
                actor=actor,
                previous_state=edge.model_dump(mode="json"),
            )

        # Remove from NetworkX
        self._graphs[graph_id].remove_node(node_id)

        # Remove from model store
        del self._nodes[graph_id][node_id]

        # Version and audit
        self._increment_version(graph_id)
        self._record_mutation(
            graph_id=graph_id,
            mutation_type=MutationType.NODE_REMOVED,
            target_id=node_id,
            actor=actor,
            previous_state=previous_state,
        )

        # Persist
        if self._config.enable_persistence and self._pool:
            await self._persist_node_remove(graph_id, node_id)

        logger.debug(
            "Removed node %s and %d edges from graph %s",
            node_id,
            len(edges_to_remove),
            graph_id,
        )

    # ------------------------------------------------------------------
    # Edge CRUD
    # ------------------------------------------------------------------

    async def add_edge(
        self,
        graph_id: str,
        source_node_id: str,
        target_node_id: str,
        commodity: str,
        quantity: Union[Decimal, float, int, str],
        edge_id: Optional[str] = None,
        actor: str = "system",
        **kwargs: Any,
    ) -> str:
        """Add a directed custody transfer edge between two nodes.

        Validates that both nodes exist and that the new edge does
        not create a cycle in the DAG. Computes a SHA-256 provenance
        hash for the edge.

        Args:
            graph_id: Target graph identifier.
            source_node_id: Upstream node (sender).
            target_node_id: Downstream node (receiver).
            commodity: EUDR commodity being transferred.
            quantity: Transfer quantity (positive).
            edge_id: Optional explicit edge ID.
            actor: Identity performing the operation.
            **kwargs: Additional SupplyChainEdge attributes.

        Returns:
            Created edge identifier.

        Raises:
            GraphNotFoundError: If graph does not exist.
            NodeNotFoundError: If either node does not exist.
            CycleDetectedError: If the edge would create a cycle.
            ValueError: If source equals target (self-loop).
        """
        self._ensure_initialized()
        self._ensure_graph_exists(graph_id)
        start = time.monotonic()

        # Validate nodes exist
        if source_node_id not in self._nodes[graph_id]:
            raise NodeNotFoundError(
                f"Source node '{source_node_id}' not found in graph {graph_id}"
            )
        if target_node_id not in self._nodes[graph_id]:
            raise NodeNotFoundError(
                f"Target node '{target_node_id}' not found in graph {graph_id}"
            )

        # Self-loop check
        if source_node_id == target_node_id:
            raise ValueError(
                f"Self-loop not allowed: node '{source_node_id}'"
            )

        # Cycle detection: check if adding this edge creates a cycle
        # A cycle exists if target can already reach source
        if self._would_create_cycle(graph_id, source_node_id, target_node_id):
            raise CycleDetectedError(
                f"Edge {source_node_id} -> {target_node_id} would create "
                f"a cycle in graph {graph_id}",
                source_node_id=source_node_id,
                target_node_id=target_node_id,
            )

        eid = edge_id or str(uuid4())

        # Normalize quantity to Decimal
        qty = Decimal(str(quantity))

        edge = SupplyChainEdge(
            edge_id=eid,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            commodity=commodity,
            quantity=qty,
            **kwargs,
        )
        edge.provenance_hash = edge.calculate_provenance_hash()

        # Add to NetworkX
        self._graphs[graph_id].add_edge(
            source_node_id,
            target_node_id,
            edge_id=eid,
            commodity=commodity,
            quantity=float(qty),
        )

        # Store model
        self._edges[graph_id][eid] = edge

        # Version and audit
        self._increment_version(graph_id)
        self._record_mutation(
            graph_id=graph_id,
            mutation_type=MutationType.EDGE_ADDED,
            target_id=eid,
            actor=actor,
            new_state=edge.model_dump(mode="json"),
        )

        # Auto-snapshot check
        await self._maybe_auto_snapshot(graph_id, actor)

        # Persist
        if self._config.enable_persistence and self._pool:
            await self._persist_edge_add(graph_id, edge)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "Added edge %s (%s -> %s, %s) to graph %s (%.1fms)",
            eid,
            source_node_id,
            target_node_id,
            commodity,
            graph_id,
            elapsed_ms,
        )
        return eid

    def get_edge(
        self, graph_id: str, edge_id: str
    ) -> SupplyChainEdge:
        """Get an edge by its identifier.

        Args:
            graph_id: Graph identifier.
            edge_id: Edge identifier.

        Returns:
            SupplyChainEdge instance.

        Raises:
            GraphNotFoundError: If graph does not exist.
            EdgeNotFoundError: If edge does not exist.
        """
        self._ensure_graph_exists(graph_id)
        edge = self._edges[graph_id].get(edge_id)
        if edge is None:
            raise EdgeNotFoundError(
                f"Edge '{edge_id}' not found in graph {graph_id}"
            )
        return edge

    def list_edges(
        self,
        graph_id: str,
        commodity: Optional[str] = None,
        source_node_id: Optional[str] = None,
        target_node_id: Optional[str] = None,
    ) -> List[SupplyChainEdge]:
        """List edges with optional filters.

        Args:
            graph_id: Graph identifier.
            commodity: Filter by commodity.
            source_node_id: Filter by source node.
            target_node_id: Filter by target node.

        Returns:
            List of matching edges.
        """
        self._ensure_graph_exists(graph_id)
        edges = list(self._edges[graph_id].values())

        if commodity is not None:
            edges = [e for e in edges if e.commodity == commodity]
        if source_node_id is not None:
            edges = [
                e for e in edges if e.source_node_id == source_node_id
            ]
        if target_node_id is not None:
            edges = [
                e for e in edges if e.target_node_id == target_node_id
            ]

        return edges

    async def remove_edge(
        self,
        graph_id: str,
        edge_id: str,
        actor: str = "system",
    ) -> None:
        """Remove an edge from the graph.

        Args:
            graph_id: Graph identifier.
            edge_id: Edge to remove.
            actor: Identity performing the operation.

        Raises:
            GraphNotFoundError: If graph does not exist.
            EdgeNotFoundError: If edge does not exist.
        """
        self._ensure_initialized()
        self._ensure_graph_exists(graph_id)

        edge = self._edges[graph_id].get(edge_id)
        if edge is None:
            raise EdgeNotFoundError(
                f"Edge '{edge_id}' not found in graph {graph_id}"
            )

        previous_state = edge.model_dump(mode="json")

        # Remove from NetworkX
        if self._graphs[graph_id].has_edge(
            edge.source_node_id, edge.target_node_id
        ):
            self._graphs[graph_id].remove_edge(
                edge.source_node_id, edge.target_node_id
            )

        # Remove from model store
        del self._edges[graph_id][edge_id]

        # Version and audit
        self._increment_version(graph_id)
        self._record_mutation(
            graph_id=graph_id,
            mutation_type=MutationType.EDGE_REMOVED,
            target_id=edge_id,
            actor=actor,
            previous_state=previous_state,
        )

        # Persist
        if self._config.enable_persistence and self._pool:
            await self._persist_edge_remove(graph_id, edge_id)

        logger.debug("Removed edge %s from graph %s", edge_id, graph_id)

    async def update_edge_attributes(
        self,
        graph_id: str,
        edge_id: str,
        actor: str = "system",
        **attributes: Any,
    ) -> SupplyChainEdge:
        """Update attributes of an existing edge.

        Args:
            graph_id: Graph identifier.
            edge_id: Edge to update.
            actor: Identity performing the operation.
            **attributes: Attribute key-value pairs to update.

        Returns:
            Updated SupplyChainEdge.

        Raises:
            GraphNotFoundError: If graph does not exist.
            EdgeNotFoundError: If edge does not exist.
        """
        self._ensure_initialized()
        self._ensure_graph_exists(graph_id)

        edge = self._edges[graph_id].get(edge_id)
        if edge is None:
            raise EdgeNotFoundError(
                f"Edge '{edge_id}' not found in graph {graph_id}"
            )

        previous_state = edge.model_dump(mode="json")

        for key, value in attributes.items():
            if hasattr(edge, key) and key not in (
                "edge_id",
                "source_node_id",
                "target_node_id",
                "created_at",
            ):
                setattr(edge, key, value)

        # Recompute provenance hash after attribute changes
        edge.provenance_hash = edge.calculate_provenance_hash()

        # Version and audit
        self._increment_version(graph_id)
        self._record_mutation(
            graph_id=graph_id,
            mutation_type=MutationType.EDGE_UPDATED,
            target_id=edge_id,
            actor=actor,
            previous_state=previous_state,
            new_state=edge.model_dump(mode="json"),
        )

        logger.debug("Updated edge %s in graph %s", edge_id, graph_id)
        return edge

    # ------------------------------------------------------------------
    # Graph algorithms
    # ------------------------------------------------------------------

    def detect_cycles(self, graph_id: str) -> List[List[str]]:
        """Detect cycles in the supply chain graph.

        EUDR supply chains must be acyclic. This method uses NetworkX
        to find all simple cycles in the directed graph.

        Args:
            graph_id: Graph to check.

        Returns:
            List of cycle paths. Empty list means no cycles (valid DAG).

        Raises:
            GraphNotFoundError: If graph does not exist.
        """
        self._ensure_graph_exists(graph_id)
        try:
            cycles = list(nx.simple_cycles(self._graphs[graph_id]))
            if cycles:
                logger.warning(
                    "Detected %d cycle(s) in graph %s",
                    len(cycles),
                    graph_id,
                )
            return cycles
        except Exception:
            # Fallback: empty graph or unexpected error
            return []

    def has_cycles(self, graph_id: str) -> bool:
        """Check whether the graph contains any cycles.

        Args:
            graph_id: Graph to check.

        Returns:
            True if cycles exist, False if the graph is a valid DAG.
        """
        self._ensure_graph_exists(graph_id)
        return not nx.is_directed_acyclic_graph(self._graphs[graph_id])

    def topological_sort(self, graph_id: str) -> List[str]:
        """Compute topological ordering of graph nodes.

        Returns nodes in an order where every node appears before
        all nodes that depend on it. Uses Kahn's algorithm via
        NetworkX for deterministic tie-breaking.

        Args:
            graph_id: Graph to sort.

        Returns:
            List of node IDs in topological order.

        Raises:
            GraphNotFoundError: If graph does not exist.
            CycleDetectedError: If graph contains cycles.
        """
        self._ensure_graph_exists(graph_id)
        graph = self._graphs[graph_id]

        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            cycle_str = " -> ".join(cycles[0]) if cycles else "unknown"
            raise CycleDetectedError(
                f"Cannot topologically sort graph {graph_id}: "
                f"cycle detected ({cycle_str})",
                cycle_path=cycles[0] if cycles else [],
            )

        return list(nx.topological_sort(graph))

    def get_ancestors(self, graph_id: str, node_id: str) -> Set[str]:
        """Get all ancestor nodes (upstream from a given node).

        Args:
            graph_id: Graph identifier.
            node_id: Target node.

        Returns:
            Set of ancestor node IDs.

        Raises:
            GraphNotFoundError: If graph does not exist.
            NodeNotFoundError: If node does not exist.
        """
        self._ensure_graph_exists(graph_id)
        if node_id not in self._nodes[graph_id]:
            raise NodeNotFoundError(
                f"Node '{node_id}' not found in graph {graph_id}"
            )
        return set(nx.ancestors(self._graphs[graph_id], node_id))

    def get_descendants(self, graph_id: str, node_id: str) -> Set[str]:
        """Get all descendant nodes (downstream from a given node).

        Args:
            graph_id: Graph identifier.
            node_id: Target node.

        Returns:
            Set of descendant node IDs.

        Raises:
            GraphNotFoundError: If graph does not exist.
            NodeNotFoundError: If node does not exist.
        """
        self._ensure_graph_exists(graph_id)
        if node_id not in self._nodes[graph_id]:
            raise NodeNotFoundError(
                f"Node '{node_id}' not found in graph {graph_id}"
            )
        return set(nx.descendants(self._graphs[graph_id], node_id))

    def get_predecessors(
        self, graph_id: str, node_id: str
    ) -> List[str]:
        """Get direct predecessor nodes (immediate upstream).

        Args:
            graph_id: Graph identifier.
            node_id: Target node.

        Returns:
            List of predecessor node IDs.
        """
        self._ensure_graph_exists(graph_id)
        if node_id not in self._nodes[graph_id]:
            raise NodeNotFoundError(
                f"Node '{node_id}' not found in graph {graph_id}"
            )
        return list(self._graphs[graph_id].predecessors(node_id))

    def get_successors(
        self, graph_id: str, node_id: str
    ) -> List[str]:
        """Get direct successor nodes (immediate downstream).

        Args:
            graph_id: Graph identifier.
            node_id: Target node.

        Returns:
            List of successor node IDs.
        """
        self._ensure_graph_exists(graph_id)
        if node_id not in self._nodes[graph_id]:
            raise NodeNotFoundError(
                f"Node '{node_id}' not found in graph {graph_id}"
            )
        return list(self._graphs[graph_id].successors(node_id))

    def get_root_nodes(self, graph_id: str) -> List[str]:
        """Get root nodes (nodes with no predecessors / in-degree 0).

        In EUDR supply chains, root nodes are typically producers
        at the origin of the supply chain.

        Args:
            graph_id: Graph identifier.

        Returns:
            Sorted list of root node IDs.
        """
        self._ensure_graph_exists(graph_id)
        graph = self._graphs[graph_id]
        return sorted(
            nid for nid in graph.nodes() if graph.in_degree(nid) == 0
        )

    def get_leaf_nodes(self, graph_id: str) -> List[str]:
        """Get leaf nodes (nodes with no successors / out-degree 0).

        In EUDR supply chains, leaf nodes are typically importers
        at the EU point-of-entry.

        Args:
            graph_id: Graph identifier.

        Returns:
            Sorted list of leaf node IDs.
        """
        self._ensure_graph_exists(graph_id)
        graph = self._graphs[graph_id]
        return sorted(
            nid for nid in graph.nodes() if graph.out_degree(nid) == 0
        )

    def get_max_depth(self, graph_id: str) -> int:
        """Calculate the maximum tier depth of the supply chain.

        Computes the longest path from any root to any leaf.

        Args:
            graph_id: Graph identifier.

        Returns:
            Maximum tier depth (0 for single-node graph).
        """
        self._ensure_graph_exists(graph_id)
        graph = self._graphs[graph_id]

        if graph.number_of_nodes() == 0:
            return 0

        if not nx.is_directed_acyclic_graph(graph):
            return -1

        return nx.dag_longest_path_length(graph)

    def get_orphan_nodes(self, graph_id: str) -> List[str]:
        """Find orphan nodes (nodes with no edges at all).

        Orphan nodes should be flagged for review per EUDR gap
        analysis requirements.

        Args:
            graph_id: Graph identifier.

        Returns:
            Sorted list of orphan (isolated) node IDs.
        """
        self._ensure_graph_exists(graph_id)
        graph = self._graphs[graph_id]
        return sorted(list(nx.isolates(graph)))

    def shortest_path(
        self, graph_id: str, source: str, target: str
    ) -> List[str]:
        """Find shortest path between two nodes.

        Args:
            graph_id: Graph identifier.
            source: Source node ID.
            target: Target node ID.

        Returns:
            List of node IDs forming the shortest path.

        Raises:
            NodeNotFoundError: If either node does not exist.
            GraphEngineError: If no path exists.
        """
        self._ensure_graph_exists(graph_id)
        for nid in (source, target):
            if nid not in self._nodes[graph_id]:
                raise NodeNotFoundError(
                    f"Node '{nid}' not found in graph {graph_id}"
                )
        try:
            return list(
                nx.shortest_path(self._graphs[graph_id], source, target)
            )
        except nx.NetworkXNoPath:
            raise GraphEngineError(
                f"No path from '{source}' to '{target}' in graph {graph_id}"
            )

    def all_paths(
        self, graph_id: str, source: str, target: str
    ) -> List[List[str]]:
        """Find all simple paths between two nodes.

        Args:
            graph_id: Graph identifier.
            source: Source node ID.
            target: Target node ID.

        Returns:
            List of paths, each path is a list of node IDs.
        """
        self._ensure_graph_exists(graph_id)
        for nid in (source, target):
            if nid not in self._nodes[graph_id]:
                raise NodeNotFoundError(
                    f"Node '{nid}' not found in graph {graph_id}"
                )
        return list(
            nx.all_simple_paths(self._graphs[graph_id], source, target)
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self, graph_id: str) -> str:
        """Serialize graph to JSON string.

        Args:
            graph_id: Graph to serialize.

        Returns:
            JSON string representation of the graph.
        """
        self._ensure_graph_exists(graph_id)
        data = self._build_export_dict(graph_id)
        return json.dumps(data, indent=2, default=str)

    @classmethod
    def from_json(
        cls,
        json_str: str,
        config: Optional[GraphEngineConfig] = None,
    ) -> Tuple["SupplyChainGraphEngine", str]:
        """Deserialize graph from JSON string.

        Creates a new engine instance with the graph loaded.

        Args:
            json_str: JSON string to parse.
            config: Optional engine configuration.

        Returns:
            Tuple of (engine, graph_id).
        """
        data = json.loads(json_str)
        engine = cls(config or GraphEngineConfig(enable_persistence=False))
        # Mark as initialized without DB
        engine._initialized = True
        graph_id = engine._load_from_dict(data)
        return engine, graph_id

    def to_graphml(self, graph_id: str) -> str:
        """Serialize graph to GraphML XML format.

        GraphML is a standard XML format for graph data interchange,
        widely supported by graph analysis tools.

        Args:
            graph_id: Graph to serialize.

        Returns:
            GraphML XML string.
        """
        self._ensure_graph_exists(graph_id)
        graph = self._graphs[graph_id]

        buf = io.BytesIO()
        nx.write_graphml(graph, buf)
        return buf.getvalue().decode("utf-8")

    @classmethod
    def from_graphml(
        cls,
        graphml_str: str,
        graph_id: Optional[str] = None,
        config: Optional[GraphEngineConfig] = None,
    ) -> Tuple["SupplyChainGraphEngine", str]:
        """Deserialize graph from GraphML XML string.

        Args:
            graphml_str: GraphML XML string.
            graph_id: Optional graph ID override.
            config: Optional engine configuration.

        Returns:
            Tuple of (engine, graph_id).
        """
        engine = cls(config or GraphEngineConfig(enable_persistence=False))
        engine._initialized = True

        gid = graph_id or str(uuid4())
        buf = io.BytesIO(graphml_str.encode("utf-8"))
        nx_graph = nx.read_graphml(buf)

        # Convert to DiGraph if not already
        if not isinstance(nx_graph, nx.DiGraph):
            nx_graph = nx.DiGraph(nx_graph)

        engine._graphs[gid] = nx_graph
        engine._nodes[gid] = {}
        engine._edges[gid] = {}
        engine._graph_versions[gid] = 1
        engine._mutation_counters[gid] = 0
        engine._audit_trail[gid] = []
        engine._latest_mutation_hash[gid] = GENESIS_HASH
        engine._graph_meta[gid] = {
            "graph_id": gid,
            "operator_id": "",
            "commodity": "",
            "graph_name": "Imported GraphML",
            "version": 1,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        # Reconstruct node models from NetworkX attributes
        for nid, attrs in nx_graph.nodes(data=True):
            node_type_str = attrs.get("node_type", "trader")
            try:
                nt = NodeType(node_type_str)
            except ValueError:
                nt = NodeType.TRADER

            engine._nodes[gid][nid] = SupplyChainNode(
                node_id=nid,
                node_type=nt,
                operator_name=attrs.get("operator_name", nid),
                country_code=attrs.get("country_code", "XX"),
            )

        return engine, gid

    def to_binary(self, graph_id: str) -> bytes:
        """Serialize graph to compact internal binary format.

        Binary format: GLSC magic (4B) + version (2B) + pickle payload.
        Used for fast graph persistence and transfer.

        Args:
            graph_id: Graph to serialize.

        Returns:
            Binary representation.
        """
        self._ensure_graph_exists(graph_id)
        data = self._build_export_dict(graph_id)

        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

        header = BINARY_MAGIC + struct.pack(">H", BINARY_VERSION)
        return header + payload

    @classmethod
    def from_binary(
        cls,
        binary_data: bytes,
        config: Optional[GraphEngineConfig] = None,
    ) -> Tuple["SupplyChainGraphEngine", str]:
        """Deserialize graph from internal binary format.

        Args:
            binary_data: Binary data to parse.
            config: Optional engine configuration.

        Returns:
            Tuple of (engine, graph_id).

        Raises:
            ValueError: If binary format is invalid.
        """
        if len(binary_data) < 6:
            raise ValueError("Binary data too short")

        magic = binary_data[:4]
        if magic != BINARY_MAGIC:
            raise ValueError(
                f"Invalid binary magic: expected {BINARY_MAGIC!r}, "
                f"got {magic!r}"
            )

        version = struct.unpack(">H", binary_data[4:6])[0]
        if version != BINARY_VERSION:
            raise ValueError(
                f"Unsupported binary version: {version}"
            )

        data = pickle.loads(binary_data[6:])  # noqa: S301

        engine = cls(config or GraphEngineConfig(enable_persistence=False))
        engine._initialized = True
        graph_id = engine._load_from_dict(data)
        return engine, graph_id

    # ------------------------------------------------------------------
    # Versioning and snapshots
    # ------------------------------------------------------------------

    async def create_snapshot(
        self,
        graph_id: str,
        created_by: str = "system",
    ) -> GraphSnapshot:
        """Create an immutable point-in-time snapshot of the graph.

        Snapshots capture the complete graph state for audit trail
        and regulatory record-keeping per EUDR Article 31.

        Args:
            graph_id: Graph to snapshot.
            created_by: Identity of the creator.

        Returns:
            GraphSnapshot with provenance hash.
        """
        self._ensure_initialized()
        self._ensure_graph_exists(graph_id)
        start = time.monotonic()

        version = self._graph_versions.get(graph_id, 1)

        # Serialize all nodes and edges
        nodes_data: Dict[str, Dict[str, Any]] = {}
        for nid, node in self._nodes[graph_id].items():
            nodes_data[nid] = node.model_dump(mode="json")

        edges_data: Dict[str, Dict[str, Any]] = {}
        for eid, edge in self._edges[graph_id].items():
            edges_data[eid] = edge.model_dump(mode="json")

        snapshot = GraphSnapshot(
            graph_id=graph_id,
            version=version,
            node_count=len(nodes_data),
            edge_count=len(edges_data),
            nodes=nodes_data,
            edges=edges_data,
            created_by=created_by,
        )
        snapshot.provenance_hash = snapshot.calculate_provenance_hash()

        # Record audit
        self._record_mutation(
            graph_id=graph_id,
            mutation_type=MutationType.GRAPH_SNAPSHOT,
            target_id=snapshot.snapshot_id,
            actor=created_by,
            new_state={
                "version": version,
                "node_count": snapshot.node_count,
                "edge_count": snapshot.edge_count,
                "provenance_hash": snapshot.provenance_hash,
            },
        )

        # Persist
        if self._config.enable_persistence and self._pool:
            await self._persist_snapshot(snapshot)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Created snapshot %s for graph %s v%d "
            "(%d nodes, %d edges, %.1fms)",
            snapshot.snapshot_id,
            graph_id,
            version,
            snapshot.node_count,
            snapshot.edge_count,
            elapsed_ms,
        )
        return snapshot

    def get_version(self, graph_id: str) -> int:
        """Get current version number of a graph.

        Args:
            graph_id: Graph identifier.

        Returns:
            Current version number.
        """
        self._ensure_graph_exists(graph_id)
        return self._graph_versions.get(graph_id, 1)

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------

    def get_audit_trail(
        self,
        graph_id: str,
        limit: int = 100,
    ) -> List[GraphMutationRecord]:
        """Get the audit trail for a graph.

        Args:
            graph_id: Graph identifier.
            limit: Maximum number of records to return.

        Returns:
            List of mutation records, most recent first.
        """
        self._ensure_graph_exists(graph_id)
        trail = self._audit_trail.get(graph_id, [])
        return list(reversed(trail[-limit:]))

    def verify_audit_chain(self, graph_id: str) -> bool:
        """Verify the integrity of the audit trail hash chain.

        Args:
            graph_id: Graph identifier.

        Returns:
            True if the hash chain is valid, False if tampered.
        """
        trail = self._audit_trail.get(graph_id, [])
        if not trail:
            return True

        prev_hash = GENESIS_HASH
        for record in trail:
            expected_hash = record.calculate_hash(prev_hash)
            if record.provenance_hash != expected_hash:
                logger.error(
                    "Audit chain broken at mutation %s: "
                    "expected %s, got %s",
                    record.mutation_id,
                    expected_hash,
                    record.provenance_hash,
                )
                return False
            prev_hash = record.provenance_hash

        logger.debug(
            "Audit chain verified for graph %s: %d records",
            graph_id,
            len(trail),
        )
        return True

    # ------------------------------------------------------------------
    # Graph statistics
    # ------------------------------------------------------------------

    def get_statistics(self, graph_id: str) -> Dict[str, Any]:
        """Compute summary statistics for a graph.

        Args:
            graph_id: Graph identifier.

        Returns:
            Dictionary of graph statistics.
        """
        self._ensure_graph_exists(graph_id)
        graph = self._graphs[graph_id]
        nodes = self._nodes[graph_id]

        # Node type distribution
        type_counts: Dict[str, int] = {}
        country_counts: Dict[str, int] = {}
        risk_counts: Dict[str, int] = {"low": 0, "standard": 0, "high": 0}

        for node in nodes.values():
            nt = node.node_type.value
            type_counts[nt] = type_counts.get(nt, 0) + 1

            cc = node.country_code
            country_counts[cc] = country_counts.get(cc, 0) + 1

            rl = node.risk_level.value
            risk_counts[rl] = risk_counts.get(rl, 0) + 1

        is_dag = nx.is_directed_acyclic_graph(graph)
        max_depth = (
            nx.dag_longest_path_length(graph) if is_dag else -1
        )

        return {
            "graph_id": graph_id,
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "is_dag": is_dag,
            "max_tier_depth": max_depth,
            "root_count": len(self.get_root_nodes(graph_id)),
            "leaf_count": len(self.get_leaf_nodes(graph_id)),
            "orphan_count": len(self.get_orphan_nodes(graph_id)),
            "node_types": type_counts,
            "countries": country_counts,
            "risk_distribution": risk_counts,
            "version": self._graph_versions.get(graph_id, 1),
            "mutation_count": self._mutation_counters.get(graph_id, 0),
        }

    # ------------------------------------------------------------------
    # Database persistence (PostgreSQL / psycopg async)
    # ------------------------------------------------------------------

    async def load_graph_from_db(self, graph_id: str) -> str:
        """Load a graph from PostgreSQL into memory.

        Reads the graph metadata, all nodes, and all edges from the
        V089 database schema and reconstructs the in-memory NetworkX
        graph.

        Args:
            graph_id: Graph identifier to load.

        Returns:
            Loaded graph identifier.

        Raises:
            PersistenceError: If database operation fails.
            GraphNotFoundError: If graph not found in database.
        """
        if not self._pool:
            raise PersistenceError("Database pool not available")

        start = time.monotonic()

        async with self._pool.connection() as conn:
            # Load graph metadata
            row = await conn.execute(
                f"SELECT * FROM {SCHEMA}.supply_chain_graphs "
                "WHERE graph_id = %s",
                (graph_id,),
            )
            graph_row = await row.fetchone()
            if not graph_row:
                raise GraphNotFoundError(
                    f"Graph '{graph_id}' not found in database"
                )

            # Initialize in-memory structures
            self._graphs[graph_id] = nx.DiGraph()
            self._nodes[graph_id] = {}
            self._edges[graph_id] = {}
            self._audit_trail[graph_id] = []
            self._latest_mutation_hash[graph_id] = GENESIS_HASH
            self._graph_versions[graph_id] = graph_row.get("version", 1)
            self._mutation_counters[graph_id] = 0

            self._graph_meta[graph_id] = {
                "graph_id": str(graph_row["graph_id"]),
                "operator_id": str(graph_row["operator_id"]),
                "commodity": graph_row["commodity"],
                "graph_name": graph_row.get("graph_name", ""),
                "version": graph_row.get("version", 1),
                "created_at": graph_row.get("created_at"),
                "updated_at": graph_row.get("updated_at"),
            }

            # Load nodes
            node_cursor = await conn.execute(
                f"SELECT * FROM {SCHEMA}.supply_chain_nodes "
                "WHERE graph_id = %s",
                (graph_id,),
            )
            node_rows = await node_cursor.fetchall()

            for nr in node_rows:
                nid = str(nr["node_id"])
                node = SupplyChainNode(
                    node_id=nid,
                    node_type=NodeType(nr["node_type"]),
                    operator_id=nr.get("operator_id"),
                    operator_name=nr["operator_name"],
                    country_code=nr["country_code"],
                    region=nr.get("region"),
                    latitude=nr.get("latitude"),
                    longitude=nr.get("longitude"),
                    commodities=(
                        json.loads(nr["commodities"])
                        if isinstance(nr.get("commodities"), str)
                        else nr.get("commodities", [])
                    ),
                    tier_depth=nr.get("tier_depth", 0),
                    risk_score=float(nr.get("risk_score", 0)),
                    risk_level=RiskLevel(
                        nr.get("risk_level", "standard")
                    ),
                    compliance_status=ComplianceStatus(
                        nr.get("compliance_status", "pending_verification")
                    ),
                    certifications=(
                        json.loads(nr["certifications"])
                        if isinstance(nr.get("certifications"), str)
                        else nr.get("certifications", [])
                    ),
                    plot_ids=(
                        json.loads(nr["plot_ids"])
                        if isinstance(nr.get("plot_ids"), str)
                        else nr.get("plot_ids", [])
                    ),
                    metadata=(
                        json.loads(nr["metadata"])
                        if isinstance(nr.get("metadata"), str)
                        else nr.get("metadata", {})
                    ),
                    created_at=nr.get(
                        "created_at", datetime.now(timezone.utc)
                    ),
                    updated_at=nr.get(
                        "updated_at", datetime.now(timezone.utc)
                    ),
                )
                self._nodes[graph_id][nid] = node
                self._graphs[graph_id].add_node(
                    nid,
                    node_type=node.node_type.value,
                    operator_name=node.operator_name,
                    country_code=node.country_code,
                )

            # Load edges
            edge_cursor = await conn.execute(
                f"SELECT * FROM {SCHEMA}.supply_chain_edges "
                "WHERE graph_id = %s",
                (graph_id,),
            )
            edge_rows = await edge_cursor.fetchall()

            for er in edge_rows:
                eid = str(er["edge_id"])
                source = str(er["source_node_id"])
                target = str(er["target_node_id"])

                edge = SupplyChainEdge(
                    edge_id=eid,
                    source_node_id=source,
                    target_node_id=target,
                    commodity=er["commodity"],
                    product_description=er.get("product_description"),
                    quantity=Decimal(str(er["quantity"])),
                    unit=er.get("unit", "kg"),
                    batch_number=er.get("batch_number"),
                    custody_model=CustodyModel(
                        er.get("custody_model", "segregated")
                    ),
                    transfer_date=er.get("transfer_date"),
                    cn_code=er.get("cn_code"),
                    hs_code=er.get("hs_code"),
                    transport_mode=er.get("transport_mode"),
                    provenance_hash=er.get("provenance_hash", ""),
                    created_at=er.get(
                        "created_at", datetime.now(timezone.utc)
                    ),
                )
                self._edges[graph_id][eid] = edge
                self._graphs[graph_id].add_edge(
                    source,
                    target,
                    edge_id=eid,
                    commodity=edge.commodity,
                    quantity=float(edge.quantity),
                )

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Loaded graph %s from DB: %d nodes, %d edges (%.1fms)",
            graph_id,
            len(self._nodes[graph_id]),
            len(self._edges[graph_id]),
            elapsed_ms,
        )
        return graph_id

    async def save_graph_to_db(self, graph_id: str) -> None:
        """Persist the complete graph state to PostgreSQL.

        Performs a full upsert of graph metadata, all nodes, and all
        edges within a single transaction for ACID consistency.

        Args:
            graph_id: Graph to persist.

        Raises:
            PersistenceError: If database operation fails.
        """
        if not self._pool:
            raise PersistenceError("Database pool not available")

        self._ensure_graph_exists(graph_id)
        start = time.monotonic()

        meta = self._graph_meta[graph_id]

        try:
            async with self._pool.connection() as conn:
                async with conn.transaction():
                    # Upsert graph
                    await conn.execute(
                        f"""
                        INSERT INTO {SCHEMA}.supply_chain_graphs
                            (graph_id, operator_id, commodity, graph_name,
                             total_nodes, total_edges, max_tier_depth,
                             traceability_score, compliance_readiness,
                             risk_summary, version, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s)
                        ON CONFLICT (graph_id) DO UPDATE SET
                            total_nodes = EXCLUDED.total_nodes,
                            total_edges = EXCLUDED.total_edges,
                            max_tier_depth = EXCLUDED.max_tier_depth,
                            version = EXCLUDED.version,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (
                            graph_id,
                            meta.get("operator_id", ""),
                            meta.get("commodity", ""),
                            meta.get("graph_name", ""),
                            len(self._nodes[graph_id]),
                            len(self._edges[graph_id]),
                            self.get_max_depth(graph_id),
                            meta.get("traceability_score", 0.0),
                            meta.get("compliance_readiness", 0.0),
                            json.dumps(
                                meta.get("risk_summary", {}),
                                default=str,
                            ),
                            self._graph_versions.get(graph_id, 1),
                            meta.get(
                                "created_at", datetime.now(timezone.utc)
                            ),
                            datetime.now(timezone.utc),
                        ),
                    )

                    # Delete existing nodes and edges, then re-insert
                    await conn.execute(
                        f"DELETE FROM {SCHEMA}.supply_chain_edges "
                        "WHERE graph_id = %s",
                        (graph_id,),
                    )
                    await conn.execute(
                        f"DELETE FROM {SCHEMA}.supply_chain_nodes "
                        "WHERE graph_id = %s",
                        (graph_id,),
                    )

                    # Insert nodes
                    for node in self._nodes[graph_id].values():
                        await self._insert_node_row(conn, graph_id, node)

                    # Insert edges
                    for edge in self._edges[graph_id].values():
                        await self._insert_edge_row(conn, graph_id, edge)

        except Exception as exc:
            raise PersistenceError(
                f"Failed to save graph {graph_id}: {exc}"
            ) from exc

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Saved graph %s to DB: %d nodes, %d edges (%.1fms)",
            graph_id,
            len(self._nodes[graph_id]),
            len(self._edges[graph_id]),
            elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_graph_exists(self, graph_id: str) -> None:
        """Verify that a graph exists in memory.

        Args:
            graph_id: Graph to check.

        Raises:
            GraphNotFoundError: If graph does not exist.
        """
        if graph_id not in self._graphs:
            raise GraphNotFoundError(
                f"Graph '{graph_id}' not found. "
                "Create it first or load from database."
            )

    def _would_create_cycle(
        self,
        graph_id: str,
        source: str,
        target: str,
    ) -> bool:
        """Check if adding edge source->target would create a cycle.

        A cycle exists if the target can already reach the source
        through existing edges (i.e., adding source->target would
        complete a cycle).

        Uses BFS from target to check reachability of source.

        Args:
            source: Proposed edge source.
            target: Proposed edge target.

        Returns:
            True if a cycle would be created.
        """
        graph = self._graphs[graph_id]

        # If target can reach source, adding source->target creates cycle
        # Use BFS for efficient reachability check
        visited: Set[str] = set()
        queue: deque[str] = deque([target])

        while queue:
            current = queue.popleft()
            if current == source:
                return True
            if current in visited:
                continue
            visited.add(current)
            for successor in graph.successors(current):
                if successor not in visited:
                    queue.append(successor)

        return False

    def _increment_version(self, graph_id: str) -> None:
        """Increment graph version and mutation counter."""
        self._graph_versions[graph_id] = (
            self._graph_versions.get(graph_id, 0) + 1
        )
        self._mutation_counters[graph_id] = (
            self._mutation_counters.get(graph_id, 0) + 1
        )
        if graph_id in self._graph_meta:
            self._graph_meta[graph_id]["updated_at"] = datetime.now(
                timezone.utc
            )

    def _record_mutation(
        self,
        graph_id: str,
        mutation_type: MutationType,
        target_id: str,
        actor: str = "system",
        previous_state: Optional[Dict[str, Any]] = None,
        new_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a graph mutation in the audit trail.

        Args:
            graph_id: Graph that was mutated.
            mutation_type: Type of mutation.
            target_id: Affected node or edge ID.
            actor: Identity of the mutator.
            previous_state: State before mutation.
            new_state: State after mutation.
        """
        if not self._config.enable_audit_trail:
            return

        prev_hash = self._latest_mutation_hash.get(
            graph_id, GENESIS_HASH
        )

        record = GraphMutationRecord(
            graph_id=graph_id,
            mutation_type=mutation_type,
            target_id=target_id,
            actor=actor,
            previous_state=previous_state,
            new_state=new_state,
        )
        record.provenance_hash = record.calculate_hash(prev_hash)

        if graph_id not in self._audit_trail:
            self._audit_trail[graph_id] = []
        self._audit_trail[graph_id].append(record)
        self._latest_mutation_hash[graph_id] = record.provenance_hash

    async def _maybe_auto_snapshot(
        self, graph_id: str, actor: str
    ) -> None:
        """Create an automatic snapshot if the mutation interval is reached.

        Args:
            graph_id: Graph identifier.
            actor: Identity for the snapshot.
        """
        if not self._config.enable_snapshots:
            return

        counter = self._mutation_counters.get(graph_id, 0)
        if counter > 0 and counter % self._config.snapshot_interval == 0:
            await self.create_snapshot(graph_id, created_by=actor)

    def _build_export_dict(self, graph_id: str) -> Dict[str, Any]:
        """Build a serializable dictionary of the full graph state.

        Args:
            graph_id: Graph to export.

        Returns:
            Complete graph state as a dictionary.
        """
        nodes_data: Dict[str, Dict[str, Any]] = {}
        for nid, node in self._nodes[graph_id].items():
            nodes_data[nid] = node.model_dump(mode="json")

        edges_data: Dict[str, Dict[str, Any]] = {}
        for eid, edge in self._edges[graph_id].items():
            edges_data[eid] = edge.model_dump(mode="json")

        meta = dict(self._graph_meta.get(graph_id, {}))
        meta["total_nodes"] = len(nodes_data)
        meta["total_edges"] = len(edges_data)

        return {
            "metadata": meta,
            "nodes": nodes_data,
            "edges": edges_data,
            "version": self._graph_versions.get(graph_id, 1),
        }

    def _load_from_dict(self, data: Dict[str, Any]) -> str:
        """Load a graph from a dictionary representation.

        Args:
            data: Dictionary with metadata, nodes, and edges.

        Returns:
            Loaded graph identifier.
        """
        meta = data.get("metadata", {})
        graph_id = meta.get("graph_id", str(uuid4()))

        self._graphs[graph_id] = nx.DiGraph()
        self._nodes[graph_id] = {}
        self._edges[graph_id] = {}
        self._graph_versions[graph_id] = data.get("version", 1)
        self._mutation_counters[graph_id] = 0
        self._audit_trail[graph_id] = []
        self._latest_mutation_hash[graph_id] = GENESIS_HASH
        self._graph_meta[graph_id] = meta

        # Load nodes
        for nid, ndata in data.get("nodes", {}).items():
            node = SupplyChainNode(**ndata)
            node.node_id = nid
            self._nodes[graph_id][nid] = node
            self._graphs[graph_id].add_node(
                nid,
                node_type=node.node_type.value,
                operator_name=node.operator_name,
                country_code=node.country_code,
            )

        # Load edges
        for eid, edata in data.get("edges", {}).items():
            # Ensure quantity is Decimal
            if "quantity" in edata:
                edata["quantity"] = Decimal(str(edata["quantity"]))
            edge = SupplyChainEdge(**edata)
            edge.edge_id = eid
            self._edges[graph_id][eid] = edge
            self._graphs[graph_id].add_edge(
                edge.source_node_id,
                edge.target_node_id,
                edge_id=eid,
                commodity=edge.commodity,
                quantity=float(edge.quantity),
            )

        logger.info(
            "Loaded graph %s: %d nodes, %d edges",
            graph_id,
            len(self._nodes[graph_id]),
            len(self._edges[graph_id]),
        )
        return graph_id

    # ------------------------------------------------------------------
    # Database persistence helpers
    # ------------------------------------------------------------------

    async def _persist_graph_create(self, graph_id: str) -> None:
        """Insert graph metadata row into PostgreSQL."""
        if not self._pool:
            return
        meta = self._graph_meta.get(graph_id, {})
        try:
            async with self._pool.connection() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {SCHEMA}.supply_chain_graphs
                        (graph_id, operator_id, commodity, graph_name,
                         version, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (graph_id) DO NOTHING
                    """,
                    (
                        graph_id,
                        meta.get("operator_id", ""),
                        meta.get("commodity", ""),
                        meta.get("graph_name", ""),
                        1,
                        meta.get("created_at", datetime.now(timezone.utc)),
                        meta.get("updated_at", datetime.now(timezone.utc)),
                    ),
                )
        except Exception as exc:
            logger.error("Failed to persist graph create: %s", exc)

    async def _persist_graph_delete(self, graph_id: str) -> None:
        """Delete graph and all related records from PostgreSQL."""
        if not self._pool:
            return
        try:
            async with self._pool.connection() as conn:
                async with conn.transaction():
                    await conn.execute(
                        f"DELETE FROM {SCHEMA}.supply_chain_edges "
                        "WHERE graph_id = %s",
                        (graph_id,),
                    )
                    await conn.execute(
                        f"DELETE FROM {SCHEMA}.supply_chain_nodes "
                        "WHERE graph_id = %s",
                        (graph_id,),
                    )
                    await conn.execute(
                        f"DELETE FROM {SCHEMA}.supply_chain_graphs "
                        "WHERE graph_id = %s",
                        (graph_id,),
                    )
        except Exception as exc:
            logger.error("Failed to persist graph delete: %s", exc)

    async def _persist_node_add(
        self, graph_id: str, node: SupplyChainNode
    ) -> None:
        """Insert a node row into PostgreSQL."""
        if not self._pool:
            return
        try:
            async with self._pool.connection() as conn:
                await self._insert_node_row(conn, graph_id, node)
        except Exception as exc:
            logger.error("Failed to persist node add: %s", exc)

    async def _persist_node_update(
        self, graph_id: str, node: SupplyChainNode
    ) -> None:
        """Update a node row in PostgreSQL."""
        if not self._pool:
            return
        try:
            async with self._pool.connection() as conn:
                db = node.to_db_dict()
                await conn.execute(
                    f"""
                    UPDATE {SCHEMA}.supply_chain_nodes SET
                        operator_name = %s, country_code = %s,
                        region = %s, latitude = %s, longitude = %s,
                        commodities = %s, tier_depth = %s,
                        risk_score = %s, risk_level = %s,
                        compliance_status = %s, certifications = %s,
                        plot_ids = %s, metadata = %s, updated_at = %s
                    WHERE node_id = %s AND graph_id = %s
                    """,
                    (
                        db["operator_name"],
                        db["country_code"],
                        db["region"],
                        db["latitude"],
                        db["longitude"],
                        db["commodities"],
                        db["tier_depth"],
                        db["risk_score"],
                        db["risk_level"],
                        db["compliance_status"],
                        db["certifications"],
                        db["plot_ids"],
                        db["metadata"],
                        db["updated_at"],
                        node.node_id,
                        graph_id,
                    ),
                )
        except Exception as exc:
            logger.error("Failed to persist node update: %s", exc)

    async def _persist_node_remove(
        self, graph_id: str, node_id: str
    ) -> None:
        """Delete a node and its edges from PostgreSQL."""
        if not self._pool:
            return
        try:
            async with self._pool.connection() as conn:
                async with conn.transaction():
                    await conn.execute(
                        f"DELETE FROM {SCHEMA}.supply_chain_edges "
                        "WHERE graph_id = %s AND "
                        "(source_node_id = %s OR target_node_id = %s)",
                        (graph_id, node_id, node_id),
                    )
                    await conn.execute(
                        f"DELETE FROM {SCHEMA}.supply_chain_nodes "
                        "WHERE node_id = %s AND graph_id = %s",
                        (node_id, graph_id),
                    )
        except Exception as exc:
            logger.error("Failed to persist node remove: %s", exc)

    async def _persist_edge_add(
        self, graph_id: str, edge: SupplyChainEdge
    ) -> None:
        """Insert an edge row into PostgreSQL."""
        if not self._pool:
            return
        try:
            async with self._pool.connection() as conn:
                await self._insert_edge_row(conn, graph_id, edge)
        except Exception as exc:
            logger.error("Failed to persist edge add: %s", exc)

    async def _persist_edge_remove(
        self, graph_id: str, edge_id: str
    ) -> None:
        """Delete an edge from PostgreSQL."""
        if not self._pool:
            return
        try:
            async with self._pool.connection() as conn:
                await conn.execute(
                    f"DELETE FROM {SCHEMA}.supply_chain_edges "
                    "WHERE edge_id = %s AND graph_id = %s",
                    (edge_id, graph_id),
                )
        except Exception as exc:
            logger.error("Failed to persist edge remove: %s", exc)

    async def _persist_snapshot(self, snapshot: GraphSnapshot) -> None:
        """Insert a snapshot row into PostgreSQL."""
        if not self._pool:
            return
        try:
            async with self._pool.connection() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {SCHEMA}.graph_snapshots
                        (snapshot_id, graph_id, version, snapshot_data,
                         provenance_hash, created_by, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        snapshot.snapshot_id,
                        snapshot.graph_id,
                        snapshot.version,
                        json.dumps(
                            {
                                "nodes": snapshot.nodes,
                                "edges": snapshot.edges,
                            },
                            default=str,
                        ),
                        snapshot.provenance_hash,
                        snapshot.created_by,
                        snapshot.created_at,
                    ),
                )
        except Exception as exc:
            logger.error("Failed to persist snapshot: %s", exc)

    async def _insert_node_row(
        self,
        conn: Any,
        graph_id: str,
        node: SupplyChainNode,
    ) -> None:
        """Insert a single node row.

        Args:
            conn: Database connection.
            graph_id: Parent graph ID.
            node: Node to insert.
        """
        db = node.to_db_dict()
        await conn.execute(
            f"""
            INSERT INTO {SCHEMA}.supply_chain_nodes
                (node_id, graph_id, node_type, operator_id, operator_name,
                 country_code, region, latitude, longitude, commodities,
                 tier_depth, risk_score, risk_level, compliance_status,
                 certifications, plot_ids, metadata, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                db["node_id"],
                graph_id,
                db["node_type"],
                db["operator_id"],
                db["operator_name"],
                db["country_code"],
                db["region"],
                db["latitude"],
                db["longitude"],
                db["commodities"],
                db["tier_depth"],
                db["risk_score"],
                db["risk_level"],
                db["compliance_status"],
                db["certifications"],
                db["plot_ids"],
                db["metadata"],
                db["created_at"],
                db["updated_at"],
            ),
        )

    async def _insert_edge_row(
        self,
        conn: Any,
        graph_id: str,
        edge: SupplyChainEdge,
    ) -> None:
        """Insert a single edge row.

        Args:
            conn: Database connection.
            graph_id: Parent graph ID.
            edge: Edge to insert.
        """
        db = edge.to_db_dict()
        await conn.execute(
            f"""
            INSERT INTO {SCHEMA}.supply_chain_edges
                (edge_id, graph_id, source_node_id, target_node_id,
                 commodity, product_description, quantity, unit,
                 batch_number, custody_model, transfer_date,
                 cn_code, hs_code, transport_mode,
                 provenance_hash, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s)
            """,
            (
                db["edge_id"],
                graph_id,
                db["source_node_id"],
                db["target_node_id"],
                db["commodity"],
                db["product_description"],
                db["quantity"],
                db["unit"],
                db["batch_number"],
                db["custody_model"],
                db["transfer_date"],
                db["cn_code"],
                db["hs_code"],
                db["transport_mode"],
                db["provenance_hash"],
                db["created_at"],
            ),
        )


# ===================================================================
# Module exports
# ===================================================================

__all__ = [
    # Engine
    "SupplyChainGraphEngine",
    # Configuration
    "GraphEngineConfig",
    # Enums
    "NodeType",
    "CustodyModel",
    "RiskLevel",
    "ComplianceStatus",
    "MutationType",
    # Data models
    "SupplyChainNode",
    "SupplyChainEdge",
    "GraphMutationRecord",
    "GraphSnapshot",
    # Exceptions
    "GraphEngineError",
    "CycleDetectedError",
    "NodeNotFoundError",
    "EdgeNotFoundError",
    "GraphNotFoundError",
    "GraphCapacityError",
    "PersistenceError",
    # Constants
    "SCHEMA",
    "GENESIS_HASH",
]
