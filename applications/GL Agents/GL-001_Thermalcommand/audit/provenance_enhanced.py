"""
Enhanced Provenance Tracker for GL-001 ThermalCommand

This module implements comprehensive provenance tracking with SHA-256 hash
chaining for complete audit trails and data lineage. It ensures zero-hallucination
compliance by maintaining cryptographic integrity of all decision inputs,
processing steps, and outputs.

Key Features:
    - SHA-256 hash chaining for tamper detection
    - Input/output hash linking for traceability
    - Model version tracking and validation
    - Full lineage graph construction
    - Merkle tree support for batch verification

Example:
    >>> tracker = EnhancedProvenanceTracker()
    >>> node = tracker.create_node(
    ...     data={"temperature": 450.5},
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

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ProvenanceNodeType(str, Enum):
    """Types of nodes in the provenance graph."""

    INPUT = "INPUT"
    TRANSFORMATION = "TRANSFORMATION"
    AGGREGATION = "AGGREGATION"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    CALCULATION = "CALCULATION"
    VALIDATION = "VALIDATION"
    OUTPUT = "OUTPUT"
    DECISION = "DECISION"
    ACTION = "ACTION"


class HashAlgorithm(str, Enum):
    """Supported cryptographic hash algorithms."""

    SHA256 = "SHA256"
    SHA384 = "SHA384"
    SHA512 = "SHA512"


class ModelVersionRecord(BaseModel):
    """Record of a model version used in processing."""

    model_id: str = Field(..., description="Unique model identifier")
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Semantic version")
    model_hash: str = Field(..., description="SHA-256 hash of model artifacts")
    framework: str = Field(..., description="ML framework (sklearn, pytorch, etc.)")
    training_timestamp: Optional[datetime] = Field(None, description="Training completion time")
    training_data_hash: Optional[str] = Field(None, description="Hash of training data")
    hyperparameters_hash: Optional[str] = Field(None, description="Hash of hyperparameters")
    validation_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Validation metrics"
    )

    class Config:
        frozen = True
        json_encoders = {datetime: lambda v: v.isoformat()}


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
    source_type: str = Field(..., description="Type of source (sensor, database, etc.)")

    # Processing information
    operation_name: Optional[str] = Field(None, description="Operation performed")
    operation_version: Optional[str] = Field(None, description="Operation version")
    operation_parameters_hash: Optional[str] = Field(None, description="Hash of parameters")

    # Model information (for inference nodes)
    model_version_record: Optional[ModelVersionRecord] = Field(
        None, description="Model version if inference"
    )

    # Chain linking
    previous_node_hash: Optional[str] = Field(None, description="Hash of previous node in chain")
    parent_node_ids: List[str] = Field(
        default_factory=list, description="Parent node IDs for graph"
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
    """
    An edge in the provenance graph representing a data flow relationship.
    """

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

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    @property
    def graph_hash(self) -> str:
        """Calculate overall graph hash."""
        # Sort nodes for deterministic hashing
        node_hashes = sorted([n.node_hash for n in self.nodes.values()])
        combined = "".join(node_hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class EnhancedProvenanceTracker:
    """
    Enhanced provenance tracker with SHA-256 chaining and full lineage support.

    This class provides comprehensive tracking of data provenance through
    the entire ThermalCommand pipeline, ensuring cryptographic integrity
    and complete auditability.

    Attributes:
        hash_algorithm: Cryptographic hash algorithm to use
        nodes: In-memory node storage (replace with DB in production)
        edges: In-memory edge storage
        chain: Ordered list of node hashes for chain verification

    Example:
        >>> tracker = EnhancedProvenanceTracker()
        >>> input_node = tracker.create_input_node(
        ...     data=sensor_data,
        ...     correlation_id="corr-123",
        ...     source_system="OPC-UA"
        ... )
        >>> output_node = tracker.create_output_node(
        ...     data=recommendations,
        ...     correlation_id="corr-123",
        ...     parent_ids=[input_node.node_id]
        ... )
        >>> graph = tracker.build_lineage_graph("corr-123", output_node.node_id)
    """

    def __init__(
        self,
        hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        storage_backend: Optional[Any] = None,
    ):
        """
        Initialize the enhanced provenance tracker.

        Args:
            hash_algorithm: Hash algorithm to use (default SHA256)
            storage_backend: Optional storage backend for persistence
        """
        self.hash_algorithm = hash_algorithm
        self.storage_backend = storage_backend

        # In-memory storage (replace with persistent storage in production)
        self._nodes: Dict[str, ProvenanceNode] = {}
        self._edges: List[ProvenanceEdge] = []
        self._chain: List[str] = []  # Hash chain
        self._model_versions: Dict[str, ModelVersionRecord] = {}

        logger.info(
            "EnhancedProvenanceTracker initialized",
            extra={"hash_algorithm": hash_algorithm.value}
        )

    def _compute_hash(self, data: Any) -> str:
        """
        Compute cryptographic hash of data.

        Args:
            data: Data to hash (will be JSON serialized)

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
        model_version: Optional[ModelVersionRecord] = None,
        parent_node_ids: Optional[List[str]] = None,
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
            model_version: Model version if inference node
            parent_node_ids: Parent node IDs in the graph
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
            model_version_record=model_version,
            previous_node_hash=previous_hash,
            parent_node_ids=parent_node_ids or [],
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

        logger.info(
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
        schema_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceNode:
        """
        Create an input provenance node.

        Args:
            data: Input data
            correlation_id: Correlation ID
            source_system: Source system (e.g., "OPC-UA", "Historian")
            source_type: Type of source
            schema_version: Schema version
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
            schema_version=schema_version,
            metadata=metadata,
        )

    def create_transformation_node(
        self,
        data: Any,
        correlation_id: str,
        operation_name: str,
        operation_version: str,
        parent_node_ids: List[str],
        operation_parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceNode:
        """
        Create a transformation provenance node.

        Args:
            data: Transformed data
            correlation_id: Correlation ID
            operation_name: Name of transformation operation
            operation_version: Version of operation
            parent_node_ids: Input node IDs
            operation_parameters: Transformation parameters
            metadata: Additional metadata

        Returns:
            Created ProvenanceNode
        """
        return self.create_node(
            data=data,
            node_type=ProvenanceNodeType.TRANSFORMATION,
            correlation_id=correlation_id,
            source_system="transformation_pipeline",
            source_type="transformation",
            operation_name=operation_name,
            operation_version=operation_version,
            operation_parameters=operation_parameters,
            parent_node_ids=parent_node_ids,
            metadata=metadata,
        )

    def create_inference_node(
        self,
        data: Any,
        correlation_id: str,
        model_version: ModelVersionRecord,
        parent_node_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceNode:
        """
        Create a model inference provenance node.

        Args:
            data: Inference output
            correlation_id: Correlation ID
            model_version: Model version record
            parent_node_ids: Input node IDs
            metadata: Additional metadata

        Returns:
            Created ProvenanceNode
        """
        # Track model version
        self._model_versions[model_version.model_id] = model_version

        return self.create_node(
            data=data,
            node_type=ProvenanceNodeType.MODEL_INFERENCE,
            correlation_id=correlation_id,
            source_system="ml_pipeline",
            source_type="model_inference",
            operation_name=f"inference:{model_version.model_name}",
            operation_version=model_version.model_version,
            model_version=model_version,
            parent_node_ids=parent_node_ids,
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

        return self.create_node(
            data=data,
            node_type=ProvenanceNodeType.CALCULATION,
            correlation_id=correlation_id,
            source_system="calculation_engine",
            source_type="deterministic_calculation",
            operation_name=calculation_name,
            operation_version=calculation_version,
            parent_node_ids=parent_node_ids,
            metadata=calc_metadata,
        )

    def create_output_node(
        self,
        data: Any,
        correlation_id: str,
        parent_node_ids: List[str],
        output_type: str = "recommendation",
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
        return self.create_node(
            data=data,
            node_type=ProvenanceNodeType.OUTPUT,
            correlation_id=correlation_id,
            source_system="output_generator",
            source_type=output_type,
            parent_node_ids=parent_node_ids,
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
        """
        Create an edge between two provenance nodes.

        Args:
            source_node_id: Source node ID
            target_node_id: Target node ID
            edge_type: Type of edge relationship
            transformation: Transformation description
            metadata: Edge metadata

        Returns:
            Created ProvenanceEdge
        """
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

    def get_node_by_hash(self, data_hash: str) -> Optional[ProvenanceNode]:
        """Get a provenance node by its data hash."""
        for node in self._nodes.values():
            if node.data_hash == data_hash:
                return node
        return None

    def get_parent_nodes(self, node_id: str) -> List[ProvenanceNode]:
        """
        Get parent nodes for a given node.

        Args:
            node_id: Node ID to get parents for

        Returns:
            List of parent ProvenanceNodes
        """
        node = self._nodes.get(node_id)
        if not node:
            return []

        parents = []
        for parent_id in node.parent_node_ids:
            parent = self._nodes.get(parent_id)
            if parent:
                parents.append(parent)
        return parents

    def get_child_nodes(self, node_id: str) -> List[ProvenanceNode]:
        """
        Get child nodes for a given node.

        Args:
            node_id: Node ID to get children for

        Returns:
            List of child ProvenanceNodes
        """
        children = []
        for node in self._nodes.values():
            if node_id in node.parent_node_ids:
                children.append(node)
        return children

    def get_nodes_by_correlation(self, correlation_id: str) -> List[ProvenanceNode]:
        """
        Get all nodes for a correlation ID.

        Args:
            correlation_id: Correlation ID to search for

        Returns:
            List of ProvenanceNodes with matching correlation ID
        """
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

        Raises:
            ValueError: If root node not found
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

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            node = self._nodes.get(current_id)
            if not node:
                continue

            graph_nodes[current_id] = node

            # Check if input node
            if node.node_type == ProvenanceNodeType.INPUT:
                input_nodes.append(current_id)

            # Add parent nodes to queue
            for parent_id in node.parent_node_ids:
                queue.append(parent_id)
                # Add edge
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
        """
        Calculate Merkle tree root hash for a set of nodes.

        Args:
            nodes: List of provenance nodes

        Returns:
            Merkle root hash
        """
        if not nodes:
            return self._compute_hash("")

        # Get leaf hashes
        hashes = sorted([node.node_hash for node in nodes])

        # Build Merkle tree
        while len(hashes) > 1:
            # Pad if odd number
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])

            # Combine pairs
            new_level = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                new_level.append(self._compute_hash(combined))
            hashes = new_level

        return hashes[0]

    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """
        Verify integrity of the hash chain.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(self._chain) < 2:
            return True, None

        # Sort nodes by timestamp to verify chain order
        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda n: n.timestamp
        )

        prev_hash = None
        for node in sorted_nodes:
            # Verify previous hash matches
            if prev_hash is not None and node.previous_node_hash != prev_hash:
                error = f"Chain broken at node {node.node_id}"
                logger.error(error)
                return False, error

            # Verify node hash is correct
            computed_hash = node.node_hash
            if computed_hash not in self._chain:
                error = f"Node hash not in chain: {node.node_id}"
                logger.error(error)
                return False, error

            prev_hash = computed_hash

        logger.info("Hash chain verified successfully")
        return True, None

    def verify_node(self, node_id: str) -> Tuple[bool, Optional[str]]:
        """
        Verify integrity of a specific node.

        Args:
            node_id: Node ID to verify

        Returns:
            Tuple of (is_valid, error_message)
        """
        node = self._nodes.get(node_id)
        if not node:
            return False, f"Node not found: {node_id}"

        # Verify node hash is in chain
        if node.node_hash not in self._chain:
            return False, f"Node hash not in chain"

        # Verify previous hash linkage
        if node.previous_node_hash:
            prev_valid = node.previous_node_hash in self._chain
            if not prev_valid:
                return False, "Previous node hash not in chain"

        return True, None

    def verify_lineage(self, graph: LineageGraph) -> Tuple[bool, Optional[str]]:
        """
        Verify integrity of a lineage graph.

        Args:
            graph: LineageGraph to verify

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Verify Merkle root
        computed_root = self._calculate_merkle_root(list(graph.nodes.values()))
        if computed_root != graph.merkle_root:
            return False, "Merkle root mismatch"

        # Verify all nodes are valid
        for node_id, node in graph.nodes.items():
            is_valid, error = self.verify_node(node_id)
            if not is_valid:
                return False, f"Node verification failed: {error}"

        # Verify graph connectivity
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

    def register_model_version(self, model_version: ModelVersionRecord) -> None:
        """
        Register a model version for tracking.

        Args:
            model_version: Model version record to register
        """
        self._model_versions[model_version.model_id] = model_version
        logger.info(
            f"Registered model version: {model_version.model_name} v{model_version.model_version}",
            extra={"model_hash": model_version.model_hash[:16] + "..."}
        )

    def get_model_version(self, model_id: str) -> Optional[ModelVersionRecord]:
        """Get registered model version by ID."""
        return self._model_versions.get(model_id)

    def get_all_model_versions(self) -> Dict[str, ModelVersionRecord]:
        """Get all registered model versions."""
        return dict(self._model_versions)

    def export_chain(self) -> List[Dict[str, Any]]:
        """
        Export the complete hash chain for external verification.

        Returns:
            List of node dictionaries with hashes
        """
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
            }
            for node in sorted_nodes
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get provenance tracking statistics.

        Returns:
            Dictionary of statistics
        """
        type_counts = {}
        for node in self._nodes.values():
            node_type = node.node_type.value
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "chain_length": len(self._chain),
            "model_versions_tracked": len(self._model_versions),
            "nodes_by_type": type_counts,
        }

    def clear(self) -> None:
        """Clear all tracked provenance data."""
        self._nodes.clear()
        self._edges.clear()
        self._chain.clear()
        self._model_versions.clear()
        logger.info("Provenance tracker cleared")
