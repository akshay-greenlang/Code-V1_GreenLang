"""
Enhanced Provenance Tracker for GL-016 Waterguard

This module implements comprehensive SHA-256 provenance tracking for
water chemistry decisions. Every decision is hashed for regulatory proof
with complete traceability including config_version, code_version, and
input_event_ids.

Key Features:
    - SHA-256 hash chaining for tamper detection
    - Input/output hash linking for traceability
    - Config and code version tracking
    - Full lineage graph construction
    - Merkle tree support for batch verification
    - Verification of provenance chain

Example:
    >>> tracker = ProvenanceTracker()
    >>> node = tracker.create_node(
    ...     data={"conductivity": 1250.5},
    ...     node_type=ProvenanceNodeType.INPUT,
    ...     source="OPC-UA"
    ... )
    >>> tracker.link_nodes(input_node.node_id, output_node.node_id)
    >>> tracker.verify_chain()
    True

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProvenanceNodeType(str, Enum):
    """Types of nodes in the provenance graph."""

    INPUT = "INPUT"
    SENSOR_READING = "SENSOR_READING"
    ANALYZER_DATA = "ANALYZER_DATA"
    TRANSFORMATION = "TRANSFORMATION"
    AGGREGATION = "AGGREGATION"
    CALCULATION = "CALCULATION"
    VALIDATION = "VALIDATION"
    RECOMMENDATION = "RECOMMENDATION"
    COMMAND = "COMMAND"
    OUTPUT = "OUTPUT"


class HashAlgorithm(str, Enum):
    """Supported cryptographic hash algorithms."""

    SHA256 = "SHA256"
    SHA384 = "SHA384"
    SHA512 = "SHA512"


class VersionInfo(BaseModel):
    """Version information for provenance tracking."""

    config_version: str = Field(..., description="Configuration version")
    code_version: str = Field(..., description="Code/agent version")
    formula_version: Optional[str] = Field(None, description="Formula/calculation version")
    model_version: Optional[str] = Field(None, description="ML model version if applicable")
    constraint_version: Optional[str] = Field(None, description="Constraint set version")

    class Config:
        frozen = True


class ProvenanceNode(BaseModel):
    """
    A node in the provenance graph representing a data transformation step.

    Each node captures the complete state of data at a point in the
    processing pipeline, including cryptographic hashes for integrity.
    """

    node_id: UUID = Field(default_factory=uuid4, description="Unique node identifier")
    correlation_id: str = Field(..., description="Correlation ID for distributed tracing")
    node_type: ProvenanceNodeType = Field(..., description="Type of provenance node")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Node creation timestamp"
    )

    # Data content
    data_hash: str = Field(..., description="SHA-256 hash of node data content")
    data_size_bytes: int = Field(..., ge=0, description="Size of data in bytes")
    schema_version: Optional[str] = Field(None, description="Schema version if applicable")

    # Source information
    source_system: str = Field(..., description="Source system identifier")
    source_type: str = Field(..., description="Type of source (sensor, analyzer, etc.)")

    # Processing information
    operation_name: Optional[str] = Field(None, description="Operation performed")
    operation_version: Optional[str] = Field(None, description="Operation version")
    operation_parameters_hash: Optional[str] = Field(None, description="Hash of parameters")

    # Version tracking for regulatory compliance
    version_info: Optional[VersionInfo] = Field(None, description="Version information")

    # Chain linking
    previous_node_hash: Optional[str] = Field(None, description="Hash of previous node in chain")
    parent_node_ids: List[str] = Field(
        default_factory=list, description="Parent node IDs for graph"
    )
    input_event_ids: List[str] = Field(
        default_factory=list, description="Input event IDs that contributed"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        frozen = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    @property
    def node_hash(self) -> str:
        """
        Calculate SHA-256 hash of node for chain linking.

        Returns:
            Hex-encoded SHA-256 hash of node content.
        """
        node_data = self.dict(exclude={"node_hash"})
        json_str = json.dumps(node_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class ProvenanceEdge(BaseModel):
    """An edge in the provenance graph representing a data flow relationship."""

    edge_id: UUID = Field(default_factory=uuid4, description="Unique edge identifier")
    source_node_id: str = Field(..., description="Source node ID")
    target_node_id: str = Field(..., description="Target node ID")
    edge_type: str = Field(..., description="Type of relationship")
    transformation: Optional[str] = Field(None, description="Transformation applied")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Edge creation timestamp"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Edge metadata")

    class Config:
        frozen = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class LineageGraph(BaseModel):
    """
    Complete lineage graph for a decision or output.

    Represents the full DAG of data transformations from inputs to outputs.
    """

    graph_id: UUID = Field(default_factory=uuid4, description="Unique graph identifier")
    correlation_id: str = Field(..., description="Correlation ID")
    root_node_id: str = Field(..., description="Root/output node ID")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Graph creation timestamp"
    )
    nodes: Dict[str, ProvenanceNode] = Field(
        default_factory=dict, description="Nodes indexed by ID"
    )
    edges: List[ProvenanceEdge] = Field(
        default_factory=list, description="Graph edges"
    )
    input_node_ids: List[str] = Field(
        default_factory=list, description="Input node IDs"
    )
    output_node_ids: List[str] = Field(
        default_factory=list, description="Output node IDs"
    )
    merkle_root: Optional[str] = Field(None, description="Merkle tree root hash")

    # Traceability information
    config_version: Optional[str] = Field(None, description="Configuration version")
    code_version: Optional[str] = Field(None, description="Code version")
    input_event_ids: List[str] = Field(
        default_factory=list, description="All input event IDs"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    @property
    def graph_hash(self) -> str:
        """Calculate overall graph hash."""
        node_hashes = sorted([n.node_hash for n in self.nodes.values()])
        combined = "".join(node_hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class DecisionProvenance(BaseModel):
    """
    Complete provenance record for a Waterguard decision.

    This is the main output for regulatory proof, containing all
    traceability information.
    """

    provenance_id: UUID = Field(default_factory=uuid4, description="Unique provenance ID")
    decision_id: str = Field(..., description="Decision/recommendation ID")
    correlation_id: str = Field(..., description="Correlation ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Provenance creation timestamp"
    )

    # Version tracking
    config_version: str = Field(..., description="Configuration version")
    code_version: str = Field(..., description="Agent code version")
    formula_version: str = Field(..., description="Formula/calculation version")
    constraint_version: str = Field(..., description="Constraint set version")

    # Input traceability
    input_event_ids: List[str] = Field(
        default_factory=list, description="All input event IDs"
    )
    input_data_hash: str = Field(..., description="Combined hash of all inputs")

    # Processing traceability
    processing_steps: List[str] = Field(
        default_factory=list, description="Processing step IDs"
    )
    calculation_hashes: Dict[str, str] = Field(
        default_factory=dict, description="Hashes of calculations performed"
    )

    # Output traceability
    output_hash: str = Field(..., description="Hash of decision output")
    output_event_id: str = Field(..., description="Output event ID")

    # Lineage
    lineage_graph_id: Optional[str] = Field(None, description="Lineage graph ID")
    merkle_root: str = Field(..., description="Merkle root of all provenance data")

    class Config:
        frozen = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    @property
    def provenance_hash(self) -> str:
        """Calculate SHA-256 hash of complete provenance."""
        data = self.dict()
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class ProvenanceTracker:
    """
    SHA-256 provenance tracker for Waterguard decisions.

    Provides comprehensive tracking of data provenance with cryptographic
    integrity for regulatory compliance and audit purposes.

    Attributes:
        hash_algorithm: Cryptographic hash algorithm to use
        nodes: In-memory node storage
        chain: Ordered list of node hashes for chain verification

    Example:
        >>> tracker = ProvenanceTracker()
        >>> input_node = tracker.create_input_node(
        ...     data=sensor_data,
        ...     correlation_id="corr-123",
        ...     source_system="OPC-UA"
        ... )
        >>> output_node = tracker.create_output_node(
        ...     data=recommendation,
        ...     correlation_id="corr-123",
        ...     parent_ids=[str(input_node.node_id)]
        ... )
        >>> graph = tracker.build_lineage_graph("corr-123", str(output_node.node_id))
    """

    def __init__(
        self,
        hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        config_version: str = "1.0.0",
        code_version: str = "1.0.0",
    ):
        """
        Initialize the provenance tracker.

        Args:
            hash_algorithm: Hash algorithm to use (default SHA256)
            config_version: Current configuration version
            code_version: Current code version
        """
        self.hash_algorithm = hash_algorithm
        self.config_version = config_version
        self.code_version = code_version

        # In-memory storage
        self._nodes: Dict[str, ProvenanceNode] = {}
        self._edges: List[ProvenanceEdge] = []
        self._chain: List[str] = []
        self._decision_provenances: Dict[str, DecisionProvenance] = {}

        logger.info(
            "ProvenanceTracker initialized",
            extra={
                "hash_algorithm": hash_algorithm.value,
                "config_version": config_version,
                "code_version": code_version,
            }
        )

    def _compute_hash(self, data: Any) -> str:
        """
        Compute cryptographic hash of data.

        Args:
            data: Data to hash

        Returns:
            Hex-encoded hash string
        """
        if isinstance(data, bytes):
            data_bytes = data
        elif isinstance(data, str):
            data_bytes = data.encode("utf-8")
        else:
            data_bytes = json.dumps(data, sort_keys=True, default=str).encode("utf-8")

        if self.hash_algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data_bytes).hexdigest()
        elif self.hash_algorithm == HashAlgorithm.SHA384:
            return hashlib.sha384(data_bytes).hexdigest()
        elif self.hash_algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512(data_bytes).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")

    def _get_chain_hash(self) -> Optional[str]:
        """Get the hash of the last node in the chain."""
        return self._chain[-1] if self._chain else None

    def create_node(
        self,
        data: Any,
        node_type: ProvenanceNodeType,
        correlation_id: str,
        source_system: str,
        source_type: str = "unknown",
        schema_version: Optional[str] = None,
        operation_name: Optional[str] = None,
        operation_version: Optional[str] = None,
        operation_parameters: Optional[Dict[str, Any]] = None,
        parent_node_ids: Optional[List[str]] = None,
        input_event_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceNode:
        """
        Create a new provenance node.

        Args:
            data: The data content for this node
            node_type: Type of provenance node
            correlation_id: Correlation ID for tracing
            source_system: Source system identifier
            source_type: Type of source
            schema_version: Schema version if applicable
            operation_name: Operation performed
            operation_version: Operation version
            operation_parameters: Parameters used in operation
            parent_node_ids: Parent node IDs in the graph
            input_event_ids: Input event IDs that contributed
            metadata: Additional metadata

        Returns:
            Created ProvenanceNode
        """
        start_time = datetime.now(timezone.utc)

        # Compute data hash
        data_hash = self._compute_hash(data)

        # Compute data size
        if isinstance(data, bytes):
            data_size = len(data)
        elif isinstance(data, str):
            data_size = len(data.encode("utf-8"))
        else:
            data_size = len(json.dumps(data, default=str).encode("utf-8"))

        # Compute parameters hash if provided
        params_hash = None
        if operation_parameters:
            params_hash = self._compute_hash(operation_parameters)

        # Get previous chain hash
        previous_hash = self._get_chain_hash()

        # Version info
        version_info = VersionInfo(
            config_version=self.config_version,
            code_version=self.code_version,
            formula_version=operation_version,
        )

        # Create node
        node = ProvenanceNode(
            correlation_id=correlation_id,
            node_type=node_type,
            data_hash=data_hash,
            data_size_bytes=data_size,
            schema_version=schema_version,
            source_system=source_system,
            source_type=source_type,
            operation_name=operation_name,
            operation_version=operation_version,
            operation_parameters_hash=params_hash,
            version_info=version_info,
            previous_node_hash=previous_hash,
            parent_node_ids=parent_node_ids or [],
            input_event_ids=input_event_ids or [],
            metadata=metadata or {},
        )

        # Store node
        node_id = str(node.node_id)
        self._nodes[node_id] = node

        # Update chain
        self._chain.append(node.node_hash)

        # Create edges for parent relationships
        for parent_id in (parent_node_ids or []):
            self._create_edge(parent_id, node_id, "derives_from")

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        logger.debug(
            f"Created provenance node: {node_type.value}",
            extra={
                "node_id": node_id,
                "correlation_id": correlation_id,
                "data_hash": data_hash[:16] + "...",
                "processing_time_ms": processing_time,
            }
        )

        return node

    def create_input_node(
        self,
        data: Any,
        correlation_id: str,
        source_system: str,
        source_type: str = "sensor",
        input_event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceNode:
        """
        Create an input provenance node.

        Args:
            data: Input data
            correlation_id: Correlation ID
            source_system: Source system
            source_type: Type of source
            input_event_id: Input event ID
            metadata: Additional metadata

        Returns:
            Created ProvenanceNode
        """
        return self.create_node(
            data=data,
            node_type=ProvenanceNodeType.INPUT,
            correlation_id=correlation_id,
            source_system=source_system,
            source_type=source_type,
            input_event_ids=[input_event_id] if input_event_id else [],
            metadata=metadata,
        )

    def create_calculation_node(
        self,
        data: Any,
        correlation_id: str,
        calculation_name: str,
        calculation_version: str,
        parent_node_ids: List[str],
        formula_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceNode:
        """
        Create a calculation provenance node (zero-hallucination).

        Args:
            data: Calculation result
            correlation_id: Correlation ID
            calculation_name: Name of calculation
            calculation_version: Version of calculation logic
            parent_node_ids: Input node IDs
            formula_hash: Hash of formula definition
            metadata: Additional metadata

        Returns:
            Created ProvenanceNode
        """
        calc_metadata = metadata or {}
        if formula_hash:
            calc_metadata["formula_hash"] = formula_hash

        # Collect input event IDs from parents
        input_event_ids = []
        for parent_id in parent_node_ids:
            parent = self._nodes.get(parent_id)
            if parent:
                input_event_ids.extend(parent.input_event_ids)

        return self.create_node(
            data=data,
            node_type=ProvenanceNodeType.CALCULATION,
            correlation_id=correlation_id,
            source_system="calculation_engine",
            source_type="deterministic_calculation",
            operation_name=calculation_name,
            operation_version=calculation_version,
            parent_node_ids=parent_node_ids,
            input_event_ids=input_event_ids,
            metadata=calc_metadata,
        )

    def create_recommendation_node(
        self,
        data: Any,
        correlation_id: str,
        recommendation_id: str,
        parent_node_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceNode:
        """
        Create a recommendation provenance node.

        Args:
            data: Recommendation data
            correlation_id: Correlation ID
            recommendation_id: Recommendation ID
            parent_node_ids: Input node IDs
            metadata: Additional metadata

        Returns:
            Created ProvenanceNode
        """
        rec_metadata = metadata or {}
        rec_metadata["recommendation_id"] = recommendation_id

        # Collect input event IDs from parents
        input_event_ids = []
        for parent_id in parent_node_ids:
            parent = self._nodes.get(parent_id)
            if parent:
                input_event_ids.extend(parent.input_event_ids)

        return self.create_node(
            data=data,
            node_type=ProvenanceNodeType.RECOMMENDATION,
            correlation_id=correlation_id,
            source_system="recommendation_engine",
            source_type="optimization",
            operation_name="generate_recommendation",
            parent_node_ids=parent_node_ids,
            input_event_ids=input_event_ids,
            metadata=rec_metadata,
        )

    def create_output_node(
        self,
        data: Any,
        correlation_id: str,
        parent_node_ids: List[str],
        output_type: str = "command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceNode:
        """
        Create an output provenance node.

        Args:
            data: Output data
            correlation_id: Correlation ID
            parent_node_ids: Parent node IDs
            output_type: Type of output
            metadata: Additional metadata

        Returns:
            Created ProvenanceNode
        """
        # Collect input event IDs from parents
        input_event_ids = []
        for parent_id in parent_node_ids:
            parent = self._nodes.get(parent_id)
            if parent:
                input_event_ids.extend(parent.input_event_ids)

        return self.create_node(
            data=data,
            node_type=ProvenanceNodeType.OUTPUT,
            correlation_id=correlation_id,
            source_system="output_generator",
            source_type=output_type,
            parent_node_ids=parent_node_ids,
            input_event_ids=list(set(input_event_ids)),  # Deduplicate
            metadata=metadata,
        )

    def _create_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_type: str,
        transformation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEdge:
        """Create an edge between two provenance nodes."""
        edge = ProvenanceEdge(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_type=edge_type,
            transformation=transformation,
            metadata=metadata or {},
        )
        self._edges.append(edge)
        return edge

    def link_nodes(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_type: str = "derives_from",
        transformation: Optional[str] = None,
    ) -> ProvenanceEdge:
        """
        Link two provenance nodes with an edge.

        Args:
            source_node_id: Source node ID
            target_node_id: Target node ID
            edge_type: Relationship type
            transformation: Transformation description

        Returns:
            Created ProvenanceEdge

        Raises:
            ValueError: If nodes don't exist
        """
        if source_node_id not in self._nodes:
            raise ValueError(f"Source node not found: {source_node_id}")
        if target_node_id not in self._nodes:
            raise ValueError(f"Target node not found: {target_node_id}")

        return self._create_edge(source_node_id, target_node_id, edge_type, transformation)

    def get_node(self, node_id: str) -> Optional[ProvenanceNode]:
        """Get a provenance node by ID."""
        return self._nodes.get(node_id)

    def get_nodes_by_correlation(self, correlation_id: str) -> List[ProvenanceNode]:
        """Get all nodes for a correlation ID."""
        return [
            node for node in self._nodes.values()
            if node.correlation_id == correlation_id
        ]

    def build_lineage_graph(
        self,
        correlation_id: str,
        root_node_id: str,
    ) -> LineageGraph:
        """
        Build complete lineage graph from root node.

        Args:
            correlation_id: Correlation ID
            root_node_id: Root/output node ID

        Returns:
            Complete LineageGraph
        """
        root_node = self._nodes.get(root_node_id)
        if not root_node:
            raise ValueError(f"Root node not found: {root_node_id}")

        # BFS to collect all ancestor nodes
        visited: Set[str] = set()
        queue = [root_node_id]
        graph_nodes: Dict[str, ProvenanceNode] = {}
        graph_edges: List[ProvenanceEdge] = []
        input_nodes: List[str] = []
        output_nodes: List[str] = [root_node_id]
        all_input_event_ids: List[str] = []

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            node = self._nodes.get(current_id)
            if not node:
                continue

            graph_nodes[current_id] = node
            all_input_event_ids.extend(node.input_event_ids)

            if node.node_type == ProvenanceNodeType.INPUT:
                input_nodes.append(current_id)

            for parent_id in node.parent_node_ids:
                queue.append(parent_id)
                graph_edges.append(ProvenanceEdge(
                    source_node_id=parent_id,
                    target_node_id=current_id,
                    edge_type="derives_from",
                ))

        # Calculate Merkle root
        merkle_root = self._calculate_merkle_root(list(graph_nodes.values()))

        graph = LineageGraph(
            correlation_id=correlation_id,
            root_node_id=root_node_id,
            nodes=graph_nodes,
            edges=graph_edges,
            input_node_ids=input_nodes,
            output_node_ids=output_nodes,
            merkle_root=merkle_root,
            config_version=self.config_version,
            code_version=self.code_version,
            input_event_ids=list(set(all_input_event_ids)),
        )

        logger.info(
            f"Built lineage graph with {len(graph_nodes)} nodes",
            extra={
                "correlation_id": correlation_id,
                "root_node_id": root_node_id,
                "merkle_root": merkle_root[:16] + "...",
            }
        )

        return graph

    def _calculate_merkle_root(self, nodes: List[ProvenanceNode]) -> str:
        """Calculate Merkle tree root hash for a set of nodes."""
        if not nodes:
            return self._compute_hash("")

        hashes = sorted([node.node_hash for node in nodes])

        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])

            new_level = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                new_level.append(self._compute_hash(combined))
            hashes = new_level

        return hashes[0]

    def create_decision_provenance(
        self,
        decision_id: str,
        correlation_id: str,
        lineage_graph: LineageGraph,
        formula_version: str = "1.0.0",
        constraint_version: str = "1.0.0",
        output_event_id: str = "",
    ) -> DecisionProvenance:
        """
        Create a complete decision provenance record.

        Args:
            decision_id: Decision/recommendation ID
            correlation_id: Correlation ID
            lineage_graph: Lineage graph for the decision
            formula_version: Formula/calculation version
            constraint_version: Constraint set version
            output_event_id: Output event ID

        Returns:
            Complete DecisionProvenance
        """
        # Compute combined input hash
        input_hashes = []
        for node_id in lineage_graph.input_node_ids:
            node = lineage_graph.nodes.get(node_id)
            if node:
                input_hashes.append(node.data_hash)
        input_data_hash = self._compute_hash("".join(sorted(input_hashes)))

        # Collect calculation hashes
        calculation_hashes = {}
        processing_steps = []
        for node_id, node in lineage_graph.nodes.items():
            if node.node_type == ProvenanceNodeType.CALCULATION:
                calculation_hashes[node.operation_name or node_id] = node.data_hash
                processing_steps.append(node_id)

        # Get output hash
        output_node = lineage_graph.nodes.get(lineage_graph.root_node_id)
        output_hash = output_node.data_hash if output_node else ""

        provenance = DecisionProvenance(
            decision_id=decision_id,
            correlation_id=correlation_id,
            config_version=self.config_version,
            code_version=self.code_version,
            formula_version=formula_version,
            constraint_version=constraint_version,
            input_event_ids=lineage_graph.input_event_ids,
            input_data_hash=input_data_hash,
            processing_steps=processing_steps,
            calculation_hashes=calculation_hashes,
            output_hash=output_hash,
            output_event_id=output_event_id,
            lineage_graph_id=str(lineage_graph.graph_id),
            merkle_root=lineage_graph.merkle_root or "",
        )

        self._decision_provenances[decision_id] = provenance

        logger.info(
            f"Created decision provenance: {decision_id}",
            extra={
                "correlation_id": correlation_id,
                "input_count": len(lineage_graph.input_event_ids),
                "merkle_root": provenance.merkle_root[:16] + "...",
            }
        )

        return provenance

    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """
        Verify integrity of the hash chain.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(self._chain) < 2:
            return True, None

        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda n: n.timestamp
        )

        prev_hash = None
        for node in sorted_nodes:
            if prev_hash is not None and node.previous_node_hash != prev_hash:
                error = f"Chain broken at node {node.node_id}"
                logger.error(error)
                return False, error

            computed_hash = node.node_hash
            if computed_hash not in self._chain:
                error = f"Node hash not in chain: {node.node_id}"
                logger.error(error)
                return False, error

            prev_hash = computed_hash

        logger.info("Hash chain verified successfully")
        return True, None

    def verify_node(self, node_id: str) -> Tuple[bool, Optional[str]]:
        """Verify integrity of a specific node."""
        node = self._nodes.get(node_id)
        if not node:
            return False, f"Node not found: {node_id}"

        if node.node_hash not in self._chain:
            return False, "Node hash not in chain"

        if node.previous_node_hash:
            if node.previous_node_hash not in self._chain:
                return False, "Previous node hash not in chain"

        return True, None

    def verify_lineage(self, graph: LineageGraph) -> Tuple[bool, Optional[str]]:
        """Verify integrity of a lineage graph."""
        computed_root = self._calculate_merkle_root(list(graph.nodes.values()))
        if computed_root != graph.merkle_root:
            return False, "Merkle root mismatch"

        for node_id in graph.nodes:
            is_valid, error = self.verify_node(node_id)
            if not is_valid:
                return False, f"Node verification failed: {error}"

        for edge in graph.edges:
            if edge.source_node_id not in graph.nodes:
                return False, f"Edge source not in graph: {edge.source_node_id}"
            if edge.target_node_id not in graph.nodes:
                return False, f"Edge target not in graph: {edge.target_node_id}"

        logger.info(
            "Lineage graph verified successfully",
            extra={"graph_id": str(graph.graph_id)}
        )
        return True, None

    def verify_decision_provenance(
        self,
        provenance: DecisionProvenance,
    ) -> Tuple[bool, Optional[str]]:
        """Verify integrity of a decision provenance record."""
        stored = self._decision_provenances.get(provenance.decision_id)
        if not stored:
            return False, "Decision provenance not found in tracker"

        if stored.provenance_hash != provenance.provenance_hash:
            return False, "Provenance hash mismatch"

        return True, None

    def get_decision_provenance(self, decision_id: str) -> Optional[DecisionProvenance]:
        """Get decision provenance by ID."""
        return self._decision_provenances.get(decision_id)

    def export_chain(self) -> List[Dict[str, Any]]:
        """Export the complete hash chain for external verification."""
        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda n: n.timestamp
        )
        return [
            {
                "node_id": str(node.node_id),
                "node_hash": node.node_hash,
                "previous_hash": node.previous_node_hash,
                "timestamp": node.timestamp.isoformat(),
                "data_hash": node.data_hash,
                "config_version": node.version_info.config_version if node.version_info else None,
                "code_version": node.version_info.code_version if node.version_info else None,
            }
            for node in sorted_nodes
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get provenance tracking statistics."""
        type_counts = {}
        for node in self._nodes.values():
            node_type = node.node_type.value
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "chain_length": len(self._chain),
            "decision_provenances": len(self._decision_provenances),
            "nodes_by_type": type_counts,
            "config_version": self.config_version,
            "code_version": self.code_version,
        }

    def clear(self) -> None:
        """Clear all tracked provenance data."""
        self._nodes.clear()
        self._edges.clear()
        self._chain.clear()
        self._decision_provenances.clear()
        logger.info("Provenance tracker cleared")
